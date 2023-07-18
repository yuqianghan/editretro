# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random, math, re, collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, NamedTuple, Optional
from torch import Tensor

from EditRetro.generate_utils.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture

from fairseq.models.transformer import Embedding
from fairseq.modules import TransformerDecoderLayer

from fairseq.models.nat import (FairseqNATModel, FairseqNATDecoder,
                                FairseqNATEncoder, ensemble_decoder,
                                ensemble_encoder)

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import LayerNorm

import numpy as np

from fairseq.models.nat.levenshtein_utils import (
    _skip,
    _skip_encoder_out,
    _fill,
    _get_advanced_ins_targets,
    _get_advanced_reposition_targets,
    _apply_ins_masks,
    _apply_ins_words,
    _apply_reposition_words,
)
from fairseq import utils

from collections import namedtuple

PretrainEncoderOut=namedtuple('PretrainEncoderOut',["encoder_out",
            "encoder_padding_mask",
            "encoder_embedding",
            "encoder_states",
            "src_tokens",
            "src_lengths",
            'masked_encoder_out',
        ])



@register_model("pretrain_mlm_editretro")
class Pretrain_MLM_EditRetro(FairseqNATModel):

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
        parser.add_argument('--pretrain', action='store_true')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = PretrainDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = PretrainEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def forward(self,
                src_tokens,
                src_lengths,
                tgt_tokens,
                masked_source=None,
                prev_output_tokens=None,
                **kwargs):

        assert tgt_tokens is not None, "forward function only supports training."
        if masked_source != None:
            masked_tokens = masked_source.ne(self.pad)
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
            lm_masked_targets = masked_source[masked_tokens]
        else:
            masked_tokens = None
            lm_masked_targets = None

        encoder_out = self.encoder(src_tokens,
                                   src_lengths=src_lengths,
                                   masked_tokens=masked_tokens,
                                   **kwargs)

        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
            prev_output_tokens, tgt_tokens, self.pad, self.unk)

        mask_ins_targets = mask_ins_targets.clamp(
            min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)

        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out)

        if masked_source == None:
            return {
                "mask_ins": {
                    "out": mask_ins_out,
                    "tgt": mask_ins_targets,
                    "mask": mask_ins_masks,
                    "ls": 0.01,
                },
                "word_ins_ml": {
                    "out": word_ins_out,
                    "tgt": tgt_tokens,
                    "mask": masked_tgt_masks,
                    "ls": self.args.label_smoothing,
                    "nll_loss": True,
                    "factor": 1.0,
                },
            }

        return {
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks,
                "ls": 0.01,
            },
            "word_ins_ml": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": 1.0,
            },
            "lm_loss": {
                'out': encoder_out.masked_encoder_out,
                'tgt': lm_masked_targets,
            }
        }

    def forward_decoder(self,
                        decoder_out,
                        encoder_out,
                        hard_constrained_decoding=False,
                        eos_penalty=0.0,
                        del_reward=0.0,
                        max_ratio=None,
                        **kwargs):

        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        # total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(
                    max_src_len)
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

            # largest element is the index till which we should compute the following
            word_reposition_pred_calc = word_reposition_pred[:, :torch.argmax(
                word_reposition_pred) + 1]
            num_deletion = word_reposition_pred_calc.eq(
                0).sum().item() - word_reposition_pred_calc.size(0)

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
            # print(decoder_out.step, "ins")
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

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_marks=output_marks,
            output_scores=output_scores,
            attn=attn,
            history=history)

    def initialize_output_tokens(self, encoder_out, src_tokens, initial_tokens,
                                 initial_marks):
        initial_tokens = initial_tokens.tolist(
        ) if initial_tokens is not None else None
        initial_marks = initial_marks.tolist(
        ) if initial_marks is not None else None
        max_num_constraints = max([len(seq) for seq in initial_tokens
                                   ]) if initial_tokens else 0
        initial_output_marks = src_tokens.new_zeros(src_tokens.size(0),
                                                    max_num_constraints + 2)
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0),
                                                     max_num_constraints + 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos

        if initial_tokens:
            for i, seq in enumerate(initial_tokens):
                for j, tok in enumerate(seq):
                    initial_output_tokens[i, j + 1] = tok
                initial_output_tokens[i, len(seq) + 1] = self.eos
                for j in range(len(seq) + 2, max_num_constraints + 2):
                    initial_output_tokens[i, j] = self.pad

        if initial_marks:
            for i, seq in enumerate(initial_marks):
                for j, mark in enumerate(seq):
                    initial_output_marks[i, j + 1] = mark

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()).type_as(
                encoder_out.encoder_out)

        return DecoderOut(output_tokens=initial_output_tokens,
                          output_marks=initial_output_marks,
                          output_scores=initial_output_scores,
                          attn=None,
                          step=0,
                          max_step=0,
                          history=None)


class PretrainEncoder(FairseqNATEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.pretrain = getattr(args, "pretrain", False)
        if self.pretrain:
            self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim,
                                                      args.encoder_embed_dim)
            self.activation_fn = utils.get_activation_fn("gelu")
            self.layer_norm1 = LayerNorm(args.encoder_embed_dim)

    def forward(self, src_tokens, src_lengths, masked_tokens=None, **kwargs):
        encoder_out = super().forward(src_tokens, src_lengths,
                                      **kwargs)  # T x B x C
        masked_encoder_out = None
        if masked_tokens is not None and self.pretrain:
            x = encoder_out.encoder_out.transpose(0, 1)  # B x T x C
            x = self.layer_norm1(
                self.activation_fn(self.lm_head_transform_weight(x)))
            x = x[masked_tokens, :]
            masked_encoder_out = F.linear(x, self.embed_tokens.weight)

        return PretrainEncoderOut(encoder_out=encoder_out.encoder_out,encoder_padding_mask= encoder_out.encoder_padding_mask,
                                  encoder_embedding=
            encoder_out.encoder_embedding,  # B x T x C
            encoder_states=
            encoder_out.encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths= [],  # B x T
            masked_encoder_out=
            masked_encoder_out if masked_tokens is not None else [])

class PretrainDecoder(FairseqNATDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args,
                         dictionary,
                         embed_tokens,
                         no_encoder_attn=no_encoder_attn)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)

        self.dropout = getattr(args, "dropout", 0.1)

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
        positions = (self.embed_positions(prev_output_tokens)
                     if self.embed_positions is not None else None)

        prev_output_tokens[prev_output_tokens >
                           self.embed_tokens.weight.shape[0]] = self.unk
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None
                and len(encoder_out.encoder_out) > 0 else None,
                encoder_out.encoder_padding_mask if
                (encoder_out is not None
                 and len(encoder_out.encoder_padding_mask) > 0) else None,
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


@register_model_architecture("pretrain_mlm_editretro",
                             "pretrain_mlm_editretro")
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

