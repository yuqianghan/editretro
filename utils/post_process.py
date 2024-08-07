from rdkit import Chem
import os
import argparse
from tqdm import tqdm
import multiprocessing
import pandas as pd
from rdkit import RDLogger
import json
import numpy as np

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


parser = argparse.ArgumentParser()
parser.add_argument('-generate_path', type=str, default='./generate.out')
parser.add_argument('-prob_path', type=str, default='')
parser.add_argument('-tgt_path', type=str, default='')
parser.add_argument('-out_path', type=str, default='')
opt = parser.parse_args()



def canonicalize_smiles_clear_map(smiles, return_max_frag=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=not opt.synthon)
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
                Chem.MolFromSmiles(smiles, sanitize=not opt.synthon)
                for smiles in sub_smi
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


assert os.path.exists(opt.generate_path) and os.path.exists(opt.prob_path) and os.path.exists(opt.tgt_path)

preds, probs = [], []

with open(opt.generate_path, 'r') as f_gen:
    preds = [pred.strip() for pred in f_gen.readlines()]
   

with open(opt.tgt_path, 'r') as f_tgt:
    tgts = [tgt.strip() for tgt in f_tgt.readlines()]
    
with open(opt.prob_path, 'r') as f_prob:
    probs = [prob.strip() for prob in f_prob.readlines()]
    

print(len(preds), len(tgts), len(probs))


### average logits for fuse predictions from different augmentations
lines = [line.strip().split(' ') for line in probs]
lines = [list(map(float, line)) for line in lines]
raw_logits = [sum(line) for line in lines]
raw_masks = [sum((np.array(line) < 0) * 1) for line in lines]
probs = [round(raw_logits[k] / raw_masks[k], 6) if raw_masks[k] > 0 else 0 for k in range(len(raw_logits))]

fout = open(opt.out_path, 'w')
for (pred, logit, tgt) in zip(preds, probs, tgts):
    print(json.dumps({
        "pred": pred,
        "logit": logit,
        "tgt": tgt,
    }), file=fout)
fout.close()