import argparse
import numpy as np
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('-generate_path', type=str, default='./generate.out')
parser.add_argument('-outpath', type=str, default='')
parser.add_argument('-predflag', type=str, default='H')
parser.add_argument('-tgt_path', type=str, default="")
opt = parser.parse_args()

fpath = opt.generate_path

tgts, preds, logits = [], [], []
for line in open(fpath, 'r'):
    if line[0] == 'T':
        i = line.split('\t')[1].strip('\n')  #.split(' ')
        tgts.append(i)
    if line[0] == opt.predflag:
        i = line.split('\t')[-1].strip('\n')
        preds.append(i)
    if line[0] == 'P':
        i = line.split('\t')[-1].strip('\n')
        logits.append(i)
print(len(tgts), len(preds), len(logits))
if len(tgts) != len(preds):
    assert os.path.exists(opt.tgt_path)
    with open(opt.tgt_path, 'r') as f:
        tgt = [''.join(i.strip().split(' ')) for i in f.readlines()]
print(len(tgt))
beam_size = int(len(preds) / len(tgt))

print(f'{beam_size=}')
tgts = []
for i in tgt:
    tgts += [i for _ in range(beam_size)]

#### recompute the logit
lines = [line.strip().split(' ') for line in logits]
lines = [list(map(float, line)) for line in lines]
raw_logits = [sum(line) for line in lines]
raw_masks = [sum((np.array(line) < 0) * 1) for line in lines]
raw_scores = [
    round(raw_logits[k] / raw_masks[k], 4) if raw_masks[k] > 0 else 0
    for k in range(len(raw_logits))
]

fout = open(opt.outpath, 'w')
for (recov, logit, ref) in zip(preds, raw_scores, tgts):
    print(json.dumps({
        "recover": recov,
        "logit": logit,
        "reference": ref,
    }),
          file=fout)
fout.close()