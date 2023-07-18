# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from EditRetro.generate_utils.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (Embedding, TransformerDecoderLayer)

from fairseq.models.nat import (FairseqNATModel, FairseqNATDecoder,
                                ensemble_decoder)

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq import utils
from EditRetro.model.levenshtein_utils import (
    _skip,
    _skip_encoder_out,
    _fill,
    _get_advanced_ins_targets,
    _get_advanced_reposition_targets,
    _apply_ins_masks,
    _apply_ins_words,
    _apply_reposition_words,
)

import pandas as pd
import numpy as np

SEED = 7777


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def same_size(t1, t2, pad_idx):
    if t1.size(1) > t2.size(1):
        pads = t2.new_full((t2.size(0), t1.size(1) - t2.size(1)), pad_idx)
        t2 = torch.cat([t2, pads], 1)
    if t1.size(1) < t2.size(1):
        pads = t1.new_full((t1.size(0), t2.size(1) - t1.size(1)), pad_idx)
        t1 = torch.cat([t1, pads], 1)
    return (t1, t2)


@register_model("editretro")
class EditRetroModel(FairseqNATModel):

    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help=
            "share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action='store_true',
            help='instead of argmax, use sampling to predict the tokens')

        parser.add_argument("--dae-ratio",
                            type=float,
                            help='ratio of noisy target as y_ins')

        parser.add_argument("--alpha-ratio",
                            type=float,
                            help='ratio of inserted string as y_del')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = EditRetroDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        decoder.dae_ratio = getattr(args, "dae_ratio", 0.5)
        decoder.alpha_ratio = getattr(args, "alpha_ratio", 0.5)
        return decoder

    def forward(self, src_tokens, src_lengths, lp_src_tokens,
                prev_output_tokens, tgt_tokens, **kwargs):

        assert tgt_tokens is not None, "forward function only supports training."

        objs = {}

        encoder_out = self.encoder(src_tokens,
                                   src_lengths=src_lengths,
                                   **kwargs)

        word_reposition = _get_advanced_reposition_targets(
            lp_src_tokens, tgt_tokens, self.pad)
        y_ins, _, _, _ = _apply_reposition_words(
            lp_src_tokens,
            None,
            None,
            None,
            word_reposition,
            self.pad,
            self.bos,
            self.eos,
        )
        if self.decoder.dae_ratio > 0:
            corrupted = (torch.rand(size=(lp_src_tokens.size(0), ),
                                    device=lp_src_tokens.device)
                         < self.decoder.dae_ratio)
            y_ins, prev_output_tokens = same_size(y_ins, prev_output_tokens,
                                                  self.pad)
            y_ins[corrupted] = prev_output_tokens[corrupted]

        y_ins, tgt_tokens = same_size(y_ins, tgt_tokens, self.pad)

        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
            y_ins, tgt_tokens, self.pad, self.unk)
        mask_ins_targets = mask_ins_targets.clamp(
            min=0, max=255)  # for safe prediction
        mask_ins_masks = y_ins[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False, prev_output_tokens=y_ins, encoder_out=encoder_out)
        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out)

        objs['mask_ins'] = {
            "out": mask_ins_out,
            "tgt": mask_ins_targets,
            "mask": mask_ins_masks,
            "ls": 0.01,
        }
        objs['word_ins'] = {
            "out": word_ins_out,
            "tgt": tgt_tokens,
            "mask": masked_tgt_masks,
            "ls": self.args.label_smoothing,
            "nll_loss": True,
            "factor": 1.0,
        }

        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)),
                1).view(word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(-1)[1]

        word_predictions.masked_scatter_(~masked_tgt_masks,
                                         tgt_tokens[~masked_tgt_masks])

        ####### string ready to learn reposition from #######
        y_del = lp_src_tokens
        if self.decoder.alpha_ratio > 0:
            y_del, word_predictions = same_size(y_del, word_predictions,
                                                self.pad)
            corrupted = (torch.rand(size=(lp_src_tokens.size(0), ),
                                    device=lp_src_tokens.device)
                         < self.decoder.alpha_ratio)
            y_del[corrupted] = word_predictions[corrupted]

        y_del, tgt_tokens = same_size(y_del, tgt_tokens, self.pad)

        word_reposition_targets = _get_advanced_reposition_targets(
            y_del, tgt_tokens, self.pad)
        word_reposition_out, _ = self.decoder.forward_word_reposition(
            normalize=False, prev_output_tokens=y_del, encoder_out=encoder_out)
        word_reposition_masks = y_del.ne(self.pad)

        objs['word_reposition'] = {
            "out": word_reposition_out,
            "tgt": word_reposition_targets,
            "mask": word_reposition_masks,
            "factor": 1.0,
        }

        return objs

    def forward_decoder(self,
                        decoder_out,
                        encoder_out,
                        src_tokens,
                        tgt_tokens,
                        hard_constrained_decoding=False,
                        eos_penalty=0.0,
                        del_reward=0.0,
                        max_ratio=None,
                        oracle_repos=False,
                        **kwargs):
        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history
        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # reposition words
        # do not apply if it is <s> </s>
        can_reposition_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_reposition_word.sum() != 0:

            word_reposition_score, word_reposition_attn = self.decoder.forward_word_reposition(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_reposition_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_reposition_word))

            if hard_constrained_decoding:
                no_del_mask = output_marks[can_reposition_word].ne(0)
                word_del_score = word_reposition_score[:, :, 0]
                word_del_score.masked_fill_(no_del_mask, -float('Inf'))
                word_reposition_score = torch.cat([
                    word_del_score.unsqueeze(2), word_reposition_score[:, :,
                                                                       1:]
                ], 2)

            word_reposition_score[:, :,
                                  0] = word_reposition_score[:, :,
                                                             0] + del_reward
            word_reposition_pred = word_reposition_score.max(-1)[1]

            num_deletion = word_reposition_pred.eq(
                0).sum().item() - word_reposition_pred.size(0)
            total_deletion_ops += num_deletion
            total_reposition_ops += word_reposition_pred.ne(
                torch.arange(word_reposition_pred.size(1),
                             device=word_reposition_pred.device).unsqueeze(
                                 0)).sum().item() - num_deletion

            if oracle_repos:
                oracle_word_reposition_pred = _get_advanced_reposition_targets(
                    output_tokens, tgt_tokens, self.pad)
                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    None,
                    oracle_word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )
            else:
                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    word_reposition_attn,
                    word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )

            output_tokens = _fill(output_tokens, can_reposition_word, _tokens,
                                  self.pad)
            output_marks = _fill(output_marks, can_reposition_word, _marks, 0)
            output_scores = _fill(output_scores, can_reposition_word, _scores,
                                  0)
            attn = _fill(attn, can_reposition_word, _attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_ins_mask))
            if eos_penalty > 0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask,
                                        None].expand_as(mask_ins_pred))

            if hard_constrained_decoding:
                no_ins_mask = output_marks[can_ins_mask][:, :-1].eq(1)
                mask_ins_pred.masked_fill_(no_ins_mask, 0)

            total_insertion_ops += mask_ins_pred.sum().item()

            _tokens, _marks, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_marks[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )

            output_tokens = _fill(output_tokens, can_ins_mask, _tokens,
                                  self.pad)
            output_marks = _fill(output_marks, can_ins_mask, _marks, 0)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_ins_word))
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens,
                                  self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_marks = output_marks[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(output_tokens=output_tokens,
                                    output_marks=output_marks,
                                    output_scores=output_scores,
                                    attn=attn,
                                    num_ops=(total_reposition_ops,
                                             total_deletion_ops,
                                             total_insertion_ops),
                                    history=history)

    def forward_decoder_step1(self,
                              decoder_out,
                              encoder_out,
                              src_tokens,
                              tgt_tokens,
                              hard_constrained_decoding=False,
                              eos_penalty=0.0,
                              del_reward=0.0,
                              max_ratio=None,
                              oracle_repos=False,
                              TOPK=5,
                              **kwargs):

        setup_seed(SEED)
        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history
        bsz = output_tokens.size(0)
        can_reposition_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_reposition_word.sum() != 0:

            word_reposition_score, word_reposition_attn = self.decoder.forward_word_reposition(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_reposition_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_reposition_word))

            if hard_constrained_decoding:
                no_del_mask = output_marks[can_reposition_word].ne(0)
                word_del_score = word_reposition_score[:, :, 0]
                word_del_score.masked_fill_(no_del_mask, -float('Inf'))
                word_reposition_score = torch.cat([
                    word_del_score.unsqueeze(2), word_reposition_score[:, :,
                                                                       1:]
                ], 2)

            word_reposition_score[:, :,
                                  0] = word_reposition_score[:, :,
                                                             0] + del_reward

            bz, SEQLEN = word_reposition_score.size(
                0), word_reposition_score.size(1)

            word_reposition_pred = word_reposition_score.topk(
                2, dim=-1)[1].permute(0, 2, 1)
            r1 = word_reposition_pred[:, 0, :].repeat_interleave(TOPK, dim=0)
            r2 = word_reposition_pred[:, 1, :].repeat_interleave(TOPK, dim=0)

            try:
                word_reposition_pred = word_reposition_score.topk(
                    5, dim=-1)[1].permute(0, 2, 1)
                N = 5
            except:
                word_reposition_pred = word_reposition_score.topk(
                    word_reposition_score.size(-1),
                    dim=-1)[1].permute(0, 2, 1)
                N = word_reposition_score.size(-1)

            r = []
            for i in range(word_reposition_pred.size(1)):
                a = word_reposition_pred[:, i, :]
                r.append(a)

            ri = torch.zeros((bz * TOPK, SEQLEN))
            for k in range(TOPK):
                for i in range(bz):
                    if k == TOPK - 1:
                        ri[i * TOPK + k, :] = r[1][i, :]
                    else:
                        ri[i * TOPK + k, :] = r[min(max(1, int(TOPK * k / N)),
                                                    len(r) - 1)][i, :]

            c = (torch.rand(size=(bz * TOPK, SEQLEN),
                            device=word_reposition_pred.device) < 0.2)
            r2[c] = ri.long().cuda(r1.device)[c]

            corrupted = torch.rand(size=(bz, TOPK, SEQLEN),
                                   device=word_reposition_pred.device)

            MIX = torch.arange(0, TOPK,
                               device=word_reposition_pred.device) / (TOPK - 1)
            corrupted = corrupted < MIX[..., None]
            corrupted = corrupted.reshape(bz * TOPK, SEQLEN)
            r1[corrupted] = r2[corrupted]
            word_reposition_pred = r1

            output_tokens = output_tokens.repeat_interleave(TOPK, dim=0)
            output_marks = output_marks.repeat_interleave(TOPK, dim=0)
            output_scores = output_scores.repeat_interleave(TOPK, dim=0)
            can_reposition_word = output_tokens.ne(self.pad).sum(1) > 2

            num_deletion = word_reposition_pred.eq(
                0).sum().item() - word_reposition_pred.size(0)
            total_deletion_ops += num_deletion
            total_reposition_ops += word_reposition_pred.ne(
                torch.arange(word_reposition_pred.size(1),
                             device=word_reposition_pred.device).unsqueeze(
                                 0)).sum().item() - num_deletion

            if oracle_repos:

                oracle_word_reposition_pred = _get_advanced_reposition_targets(
                    output_tokens, tgt_tokens, self.pad)
                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    None,
                    oracle_word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )
            else:
                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    word_reposition_attn,
                    word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )
            output_tokens = _fill(output_tokens, can_reposition_word, _tokens,
                                  self.pad)
            output_marks = _fill(output_marks, can_reposition_word, _marks, 0)
            output_scores = _fill(output_scores, can_reposition_word, _scores,
                                  0)
            attn = _fill(attn, can_reposition_word, _attn, 0.)
            if history is not None:
                history.append(output_tokens.clone().repeat_interleave(2,
                                                                       dim=0))
        return decoder_out._replace(output_tokens=output_tokens,
                                    output_marks=output_marks,
                                    output_scores=output_scores,
                                    attn=attn,
                                    num_ops=(total_reposition_ops,
                                             total_deletion_ops,
                                             total_insertion_ops),
                                    history=history)

    def forward_decoder_step2(self,
                              decoder_out,
                              encoder_out,
                              src_tokens,
                              tgt_tokens,
                              hard_constrained_decoding=False,
                              eos_penalty=0.0,
                              del_reward=0.0,
                              max_ratio=None,
                              oracle_repos=False,
                              TOPK=2,
                              **kwargs):
        setup_seed(SEED)
        # TOPK = 2
        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history
        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_ins_mask))
            if eos_penalty > 0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask,
                                        None].expand_as(mask_ins_pred))

            if hard_constrained_decoding:
                no_ins_mask = output_marks[can_ins_mask][:, :-1].eq(1)
                mask_ins_pred.masked_fill_(no_ins_mask, 0)

            total_insertion_ops += mask_ins_pred.sum().item()

            _tokens, _marks, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_marks[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )

            output_tokens = _fill(output_tokens, can_ins_mask, _tokens,
                                  self.pad)
            output_marks = _fill(output_marks, can_ins_mask, _marks, 0)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone().repeat_interleave(2,
                                                                       dim=0))

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_ins_word))
            word_ins_score, word_ins_pred = word_ins_score.topk(TOPK, dim=-1)

            SEQLEN = word_ins_score.size(1)

            word_ins_score = word_ins_score.permute(0, 2,
                                                    1).reshape(-1, SEQLEN)
            word_ins_pred = word_ins_pred.permute(0, 2, 1).reshape(-1, SEQLEN)

            output_tokens = output_tokens.repeat_interleave(TOPK, dim=0)
            output_marks = output_marks.repeat_interleave(TOPK, dim=0)
            output_scores = output_scores.repeat_interleave(TOPK, dim=0)
            if attn != None:
                attn = attn.repeat_interleave(TOPK, dim=0)

            can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
            # word_ins_score, word_ins_pred = word_ins_score.max(-1)

            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens,
                                  self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        else:
            output_tokens = output_tokens.repeat_interleave(TOPK, dim=0)
            output_marks = output_marks.repeat_interleave(TOPK, dim=0)
            output_scores = output_scores.repeat_interleave(TOPK, dim=0)
            if attn != None:
                attn = attn.repeat_interleave(TOPK, dim=0)
            can_ins_word = output_tokens.eq(self.unk).sum(1) > 0

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_marks = output_marks[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(output_tokens=output_tokens,
                                    output_marks=output_marks,
                                    output_scores=output_scores,
                                    attn=attn,
                                    num_ops=(total_reposition_ops,
                                             total_deletion_ops,
                                             total_insertion_ops),
                                    history=history)

        # insert placeholders

    def initialize_output_tokens(self,
                                 encoder_out,
                                 src_tokens,
                                 init_tokens=None):
        if init_tokens is not None:
            initial_output_tokens = init_tokens
        else:
            initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens[:, 1] = self.eos

        initial_output_marks = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()).type_as(encoder_out.encoder_out)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()).type_as(encoder_out.encoder_out)

        return DecoderOut(output_tokens=initial_output_tokens,
                          output_marks=initial_output_marks,
                          output_scores=initial_output_scores,
                          attn=None,
                          step=0,
                          max_step=0,
                          num_ops=(0, 0, 0),
                          history=None)


class EditRetroDecoder(FairseqNATDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args,
                         dictionary,
                         embed_tokens,
                         no_encoder_attn=no_encoder_attn)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion",
                                             False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(',')]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(self.early_exit[1])
            ])
        self.layers_reposition = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_reposition = nn.ModuleList([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(self.early_exit[0])
            ])

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(args, "no_share_discriminator",
                           False), "must set saperate discriminator"
            self.layers_msk = self.layers_reposition

    def extract_features(self,
                         prev_output_tokens,
                         encoder_out=None,
                         early_exit=None,
                         layers=None,
                         **unused):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the EDITORTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (self.embed_positions(prev_output_tokens)
                     if self.embed_positions is not None else None)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask
                if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens,
                         **unused):
        features, extra = self.extract_features(prev_output_tokens,
                                                encoder_out=encoder_out,
                                                early_exit=self.early_exit[1],
                                                layers=self.layers_msk,
                                                **unused)
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens,
                         **unused):
        features, extra = self.extract_features(prev_output_tokens,
                                                encoder_out=encoder_out,
                                                early_exit=self.early_exit[2],
                                                layers=self.layers,
                                                **unused)
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_reposition(self, normalize, encoder_out,
                                prev_output_tokens, **unused):
        features, extra = self.extract_features(prev_output_tokens,
                                                encoder_out=encoder_out,
                                                early_exit=self.early_exit[2],
                                                layers=self.layers_reposition,
                                                **unused)
        prev_output_embed = self.embed_tokens(prev_output_tokens)
        # B x T x T
        decoder_out = torch.bmm(features, prev_output_embed.transpose(1, 2))
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']


@register_model_architecture("editretro", "editretro")
def EditRetro_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before",
                                            False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim",
                                     args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim",
                                         args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before",
                                            False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff",
                                           None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout",
                                            0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim",
                                      args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim",
                                     args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator",
                                          False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor",
                                          False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)
