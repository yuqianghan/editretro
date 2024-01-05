from rdkit import Chem
import os
import argparse
import multiprocessing
from rdkit import RDLogger
import re
import numpy as np

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def canonicalize_smiles_clear_map(smiles, return_max_frag=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        [
            atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()
            if atom.HasProp('molAtomMapNumber')
        ]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return '', ''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [
                Chem.MolFromSmiles(smiles, sanitize=True) for smiles in sub_smi
            ]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms()))
                            for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(
                    sorted(sub_mol_size, key=lambda x: x[1],
                           reverse=True)[0][0],
                    return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '', ''
        else:
            return ''


def compute_rank(prediction, score, alpha=1.0, beam_size=10):
    valid_score = [[k for k in range(len(prediction[j]))]
                   for j in range(len(prediction))]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    max_frag_rank = {}
    highest = {}

    for j in range(len(prediction)):
        for k in range(len(prediction[j])):
            if prediction[j][k][0] == "":
                valid_score[j][k] = beam_size + 1
                invalid_rates[k] += 1
        de_error = [
            i[0] for i in sorted(list(zip(prediction[j], score[j])),
                                 key=lambda x: x[1],
                                 reverse=True) if i[0][0] != ""
        ]
        de_score = [
            i[1] for i in sorted(list(zip(prediction[j], score[j])),
                                 key=lambda x: x[1],
                                 reverse=True) if i[0][0] != ""
        ]
        prediction[j] = de_error
        score[j] = de_score

        for k, data in enumerate(prediction[j]):
            if data in rank:
                rank[data] += 1 / (alpha * k + 1)
            else:
                rank[data] = 1 / (alpha * k + 1)
            if data in highest:
                highest[data] = min(k, highest[data])
            else:
                highest[data] = k

    return rank, invalid_rates


def main_rank(raw_predictions, raw_scores, steps, opt):
    data_size = len(raw_predictions) // (
        opt.augmentation * opt.beam_size) if opt.length == -1 else opt.length
    predictions = raw_predictions[:data_size *
                                  (opt.augmentation * opt.beam_size)]

    pool = multiprocessing.Pool(processes=opt.process_number)
    predictions1 = pool.map(func=canonicalize_smiles_clear_map,
                            iterable=predictions)
    pool.close()
    pool.join()

    predictions = [[[] for j in range(opt.augmentation)]
                   for i in range(data_size)
                   ]  # data_len x augmentation x beam_size
    for i, line in enumerate(predictions1):
        predictions[i //
                    (opt.beam_size *
                     opt.augmentation)][i %
                                        (opt.beam_size * opt.augmentation) //
                                        opt.beam_size].append(line)

    raw_scores = raw_scores[:data_size * (opt.augmentation * opt.beam_size)]
    scores = [[[] for j in range(opt.augmentation)]
              for i in range(data_size)]  # data_len x augmentation x beam_size
    for i, line in enumerate(raw_scores):
        scores[i //
               (opt.beam_size * opt.augmentation)][i % (opt.beam_size *
                                                        opt.augmentation) //
                                                   opt.beam_size].append(line)
    data_size = len(predictions)
    print("data size ", data_size)

    ranked_results = []

    for i in range(len(predictions)):
        rank, _ = compute_rank(predictions[i],
                               scores[i],
                               alpha=opt.score_alpha,
                               beam_size=opt.beam_size)
        rank = list(zip(rank.keys(), rank.values()))
        rank.sort(key=lambda x: x[1], reverse=True)
        rank = rank[:opt.n_best]
        ranked_results.append([item[0][0] for item in rank])

    if opt.save_file != "":
        with open(opt.save_file, "w") as f:
            for res in ranked_results:
                for smi in res:
                    f.write(smi)
                    f.write("\n")
                f.write("\n")

        print(f'The result is saved in {opt.save_file}')

    if opt.output_edit_step:
        top1 = [i[0] for i in ranked_results]
        output_step = []
        k = opt.beam_size * opt.augmentation

        for i, smi in enumerate(top1):
            flag = False
            for j in range(k):
                try:
                    if Chem.CanonSmiles(
                            raw_predictions[i * k +
                                            j]) == Chem.CanonSmiles(smi):
                        output_step.append(steps[i * k + j])
                        flag = True
                        break
                except:
                    pass
            if not flag:
                output_step.append(['invalid predictions'])

        with open(opt.save_file, "a") as f:
            f.write('top1 steps:\n')
            for s in output_step:
                for ss in s:
                    f.write(str(ss))
                f.write('\n')

    return ranked_results


def process_input(fpath, output_edit_step):
    b, c = [], []
    for line in open(fpath, 'r'):
        if line[0] == 'H':
            i = line.split('\t')[-1].strip('\n')
            b.append(i)
        if line[0] == 'P':
            i = line.split('\t')[-1].strip('\n')
            c.append(i)
    print(len(b), len(c))

    steps = None
    if output_edit_step:
        newstep, steps = [], []
        for i, line in enumerate(open(fpath, 'r')):
            if line[0] == 'O':
                newstep = []
            if line[0] == 'E':
                a = 'E-' + line.split('\t')[0].split('_')[-1] + '\t'
                a = a + line.split('\t')[-1]
                newstep.append(a)
            if line[0] == 'S' and newstep != []:
                steps.append(newstep)
        steps.append(newstep)

    lines = [line.strip().split(' ') for line in c]
    lines = [list(map(float, line)) for line in lines]
    raw_logits = [sum(line) for line in lines]
    raw_masks = [sum((np.array(line) < 0) * 1) for line in lines]
    raw_scores = [
        round(raw_logits[k] / raw_masks[k], 4) if raw_masks[k] > 0 else 0
        for k in range(len(raw_logits))
    ]

    predictions = [''.join(i.split(' ')) for i in b]

    return predictions, raw_scores, steps


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='get_ranked_topk.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-beam_size', type=int, default=1, help='Beam size')
    parser.add_argument('-n_best', type=int, default=1, help='n best')
    parser.add_argument('-output_file',
                        type=str,
                        required=True,
                        help="fairseq output log file")
    parser.add_argument('-augmentation', type=int, default=10)
    parser.add_argument('-score_alpha', type=float, default=0.1)
    parser.add_argument('-length', type=int, default=-1)
    parser.add_argument('-process_number',
                        type=int,
                        default=multiprocessing.cpu_count())
    parser.add_argument('-save_file',
                        type=str,
                        default='./output_ranked_topk.txt')
    parser.add_argument('-output_edit_step', action='store_true')

    opt = parser.parse_args()

    predictions, raw_scores, steps = process_input(opt.output_file,
                                                   opt.output_edit_step)

    main_rank(predictions, raw_scores, steps, opt)