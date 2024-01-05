# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import os
import random

import torch

from fairseq.data import noising
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask  #, load_langpair_dataset
from fairseq import utils

from fairseq.data import (
    MaskTokensDataset,
    AppendTokenDataset,
    ConcatDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)

from fairseq.tasks import register_task

from editretro.data.masked_pair_dataset import LanguagePairDataset
from editretro.data.mask_tokens_dataset  import MaskTokensDataset


@register_task('translation_pretrain')
class JointlyMaskedMLMTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--noise',
                            default='random_delete',
                            choices=[
                                'random_delete', 'random_shuffle',
                                'random_delete_shuffle', 'random_mask',
                                'no_noise', 'full_mask'
                            ])
        parser.add_argument('--random-seed', default=1, type=int)
        parser.add_argument('--warmup-iters', default=5000, type=int)
        parser.add_argument('--decay-schedule', default=0.0, type=float)

        # mask args
        parser.add_argument('--mask-prob',default=0.15,type=float)
        parser.add_argument('--leave-unmasked-prob',default=0.1,type=float)
        parser.add_argument('--random-token-prob',default=0.1,type=float)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args,src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.label_dictionary = None

        # Add mask token for both souce dict and target dict.

        if args.mask_prob>0.0:
            if '<mask>' not in src_dict.symbols:
                self.mask_idx = src_dict.add_symbol("<mask>")
                tgt_dict.add_symbol('<mask>')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            shuffle=(split != "test"),
            mask_idx=self.mask_idx,
            mask_prob = self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            )
    def inject_noise(self, target_tokens):

        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_shuffle(target_tokens, p, max_shuffle_distance):
            word_shuffle = noising.WordShuffle(self.tgt_dict)
            target_mask = target_tokens.eq(self.tgt_dict.pad())
            target_length = target_mask.size(1) - target_mask.long().sum(1)
            prev_target_tokens, _ = word_shuffle.noising(
                target_tokens.t().cpu(), target_length.cpu(),
                max_shuffle_distance)
            prev_target_tokens = prev_target_tokens.to(
                target_tokens.device).t()
            masks = (target_tokens.clone().sum(
                dim=1, keepdim=True).float().uniform_(0, 1) < p)
            prev_target_tokens = masks * prev_target_tokens + (
                ~masks) * target_tokens
            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(
                target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete_shuffle':
            return _random_shuffle(_random_delete(target_tokens),
                                   self.args.delete_prob, 3)
        elif self.args.noise == 'random_shuffle':
            return _random_shuffle(target_tokens, self.args.delete_prob, 3)
        elif self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_dataset_for_inference(self, src_tokens, src_lengths, target_tokens,target_lengths,num_source_inputs,constraints=None):
        if num_source_inputs==1:
            from fairseq.data.language_pair_dataset import LanguagePairDataset
            return LanguagePairDataset(
                src_tokens,
                src_lengths,
                self.source_dictionary,
                tgt=target_tokens,
                tgt_sizes=target_lengths,
                tgt_dict=self.target_dictionary,
                shuffle=False,
                append_bos=True,
                input_feeding=False
            )
        else:
            raise ValueError( "num_source_inputs!=1" )

    def build_generator(self, args):
        from editretro.models.iterative_refinement_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            del_reward=getattr(args, 'iter_decode_deletion_reward', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker',
                              False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            constrained_decoding=getattr(args, 'constrained_decoding', False),)

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()

        sample['prev_target'] = self.inject_noise(sample['target'])
        loss, sample_size, logging_output = criterion(model,
                                                      sample)

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'])
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=True,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    shuffle=True,
    # Masked LM parameters.
    mask_idx: int = 0,
    seed: int = 1,
    mask_prob: float = 0.01,
    leave_unmasked_prob: float = 0.0,
    random_token_prob: float = 0.0,
    freq_weighted_replacement: bool = False,
    mask_whole_words: torch.Tensor = None,
    mask_multiple_length: int = 1,
    mask_stdev: float = 0.0,
    label_dataset = None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        if not combine:
            break
    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(
            tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(
            data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    # mask source dataset.
    src_dataset, masked_src_dataset = MaskTokensDataset.apply_mask(
        src_dataset,
        src_dict,
        pad_idx=src_dict.pad(),
        mask_idx=mask_idx,
        seed=seed,
        mask_prob=mask_prob,
        leave_unmasked_prob=leave_unmasked_prob,
        random_token_prob=random_token_prob,
        freq_weighted_replacement=freq_weighted_replacement,
        mask_whole_words=mask_whole_words,
        mask_multiple_length=mask_multiple_length,
        mask_stdev=mask_stdev,
    )
    
    # for Mask Prediction calculation
    # Use random delete in inject_noise to add noise to target tokens
    masked_tgt_dataset=None

    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        # for Mask LM loss calculation.
        masked_src_dataset,
        masked_src_dataset.sizes,
        # for Mask Prediction calculation
        masked_tgt_dataset,
        None,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        shuffle=shuffle,
        input_feeding=False,
        label=label_dataset,
    )
