#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch

from fairseq import bleu, checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.logging.meters import StopwatchMeter

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')

Batch = namedtuple('Batch', 'ids src_tokens src_lengths tgt_tokens decoder_input')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

from rdkit import Chem
import codecs
from SmilesPE.tokenizer import *
import re

spe_vob = codecs.open('./preprocess/SPE_ChEMBL.txt')
spe_tokenizer = SPE_Tokenizer(spe_vob)


def smi_tokenizer(smi, spe=True):
    if spe:
        return spe_tokenizer.tokenize(smi)
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def get_aug_input(smiles, aug=10):
    mol = Chem.MolFromSmiles(smiles)
    a = [smi_tokenizer(smiles)]

    while len(a) < aug:
        t = Chem.MolToSmiles(mol, doRandom=True)
        a.append(smi_tokenizer(t))
    return a


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn, separator=' <sep> '):

    # if args.constrained_decoding:
    if args.inference_with_augmentation:
        num_source_inputs = 1

        aug_line = []
        for i in lines:
            aug_line += get_aug_input(i, args.aug)
        lines = aug_line
        tokens = [
            task.source_dictionary.encode_line(encode_fn(src_str.split(separator)[0]), add_if_not_exist=False).long()
            for src_str in lines
        ]
        lengths = [t.numel() for t in tokens]

    else:
        num_source_inputs = 1
        tokens = [
            task.source_dictionary.encode_line(encode_fn(src_str.split(separator)[0]), add_if_not_exist=False).long()
            for src_str in lines
        ]
        lengths = [t.numel() for t in tokens]

    target_tokens = None
    target_lengths = None
    if args.has_target:
        target_tokens = [
            task.target_dictionary.encode_line(encode_fn(src_str.split(separator)[-1]), add_if_not_exist=False).long()
            for src_str in lines
        ]
        target_lengths = [t.numel() for t in target_tokens]

    itr = task.get_batch_iterator(dataset=task.build_dataset_for_inference(tokens, lengths, target_tokens, target_lengths,
                                                                           num_source_inputs),
                                  max_tokens=args.max_tokens,
                                  max_sentences=args.max_sentences,
                                  max_positions=max_positions,
                                  ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(ids=batch['id'],
                    src_tokens=batch['net_input']['src_tokens'],
                    src_lengths=batch['net_input']['src_lengths'],
                    tgt_tokens=batch['target'],
                    decoder_input=batch['decoder_input'])


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(task.max_positions(), *[model.max_positions() for model in models])

    num_sentences = 0
    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths

            tgt_tokens = batch.tgt_tokens
            num_sentences += src_tokens[0].size(0)
            decoder_input = batch.decoder_input
            if use_cuda:
                if isinstance(src_tokens, list):
                    src_tokens = [tokens.cuda() for tokens in src_tokens]
                    src_lengths = [lengths.cuda() for lengths in src_lengths]
                else:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()
                    if decoder_input is not None:
                        decoder_input = decoder_input.cuda()

                if tgt_tokens is not None:
                    tgt_tokens = tgt_tokens.cuda()

            if tgt_tokens is not None:
                sample = {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_lengths': src_lengths,
                    },
                    'target': tgt_tokens,
                    'decoder_input': decoder_input,
                }
            else:
                sample = {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_lengths': src_lengths,
                    },
                    'decoder_input': decoder_input,
                }

            gen_timer.start()
            translations = task.inference_step(generator, models, sample)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in translations)
            gen_timer.stop(num_generated_tokens)

            bid = [val for val in batch.ids.tolist() for i in range(task.topk)]

            src_tokens = src_tokens.repeat_interleave(task.topk, dim=0)
            if tgt_tokens is not None:
                tgt_tokens = tgt_tokens.repeat_interleave(task.topk, dim=0)

            for i, (id, hypos) in enumerate(zip(bid, translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                tgt_tokens_i = None
                if tgt_tokens is not None:
                    tgt_tokens_i = utils.strip_pad(tgt_tokens[i, :], tgt_dict.pad()).int().cpu()
                results.append((start_id + id, src_tokens_i, hypos, tgt_tokens_i))

        # sort output to match input order
        for id, src_tokens, hypos, tgt_tokens in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            if tgt_tokens is not None:
                tgt_str = tgt_dict.string(tgt_tokens, args.remove_bpe, escape_unk=True)
                print('T-{}\t{}'.format(id, tgt_str))

            # Process top predictions
            for j, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2
                print('H-{}\t{}\t{}'.format(id, score, hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(
                        map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))))
                if args.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(id, alignment_str))
                if args.print_step:
                    print('I-{}\t{}'.format(id, hypo['steps']))
                    print('O-{}\t{}'.format(id, hypo['num_ops']))

                if getattr(args, 'retain_iter_history', False):
                    for step, h in enumerate(hypo['history']):
                        _, h_str, _ = utils.post_process_prediction(
                            hypo_tokens=h['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=None,
                            align_dict=None,
                            tgt_dict=tgt_dict,
                            remove_bpe=None,
                        )
                        print('E-{}_{}\t{}'.format(id, step, h_str))

                # Score only the top hypothesis
                if tgt_tokens is not None and j == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        tgt_tokens = tgt_dict.encode_line(tgt_str, add_if_not_exist=True)
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(tgt_str, hypo_str)
                    else:
                        scorer.add(tgt_tokens, hypo_tokens)

        # update running id counter
        start_id += len(inputs)

    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if args.has_target:
        logger.info('Generate with beam={}: {}'.format(args.beam, scorer.result_string()))


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
