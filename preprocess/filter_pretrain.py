from collections import Counter
import numpy as np
import tqdm
import os
import shutil


def main():

    test_file = '../datasets/USPTO_50K/aug20/test.src'
    input_path = '../datasets/USPTO_FULL/aug10/'
    output_path = '../datasets/USPTO_Pretrain/pretrain/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    shutil.copyfile(input_path + 'test.src', output_path + 'test.src')
    shutil.copyfile(input_path + 'test.tgt', output_path + 'test.tgt')
    shutil.copyfile(input_path + 'val.src', output_path + 'val.src')
    shutil.copyfile(input_path + 'val.tgt', output_path + 'val.tgt')
    
    test_aug = 20
    train_aug = 10

    test_mols = []

    with open(test_file, 'r') as f1:
        lines = f1.readlines()
        test_mols.extend([''.join(lines[i].strip('\n').split(' ')) for i in range(0, len(lines), test_aug)])
    
    print(len(test_mols))
    print(len(list(set(test_mols))))
    
    test_mols = set(test_mols)
    
    filtered_src, filtered_tgt = [], []
    
    with open(input_path + 'train.src', 'r') as f3:
        train_src = f3.readlines()
        src_mols = [''.join(train_src[i].strip('\n').split(' ')) for i in range(0, len(train_src), train_aug)]
    
    with open(input_path + 'train.tgt', 'r') as f4:
        train_tgt = f4.readlines()
        tgt_mols = []
        for i in range(0, len(train_tgt), train_aug):
            mols = train_tgt[i].split('.')
            tgt_mols.append([''.join(mol.strip('\n').split(' ')) for mol in mols])    
        
    for i in range(len(src_mols)):    
        if src_mols[i] not in test_mols and not set(tgt_mols[i]).intersection(test_mols):
            filtered_src.extend(train_src[i * train_aug : (i+1) * train_aug])
            filtered_tgt.extend(train_tgt[i * train_aug : (i+1) * train_aug])
                
    print(len(train_src), len(filtered_src))
    print(len(train_tgt), len(filtered_tgt))
    
    with open(output_path + "train.src", 'w') as fsrc:
        for k in filtered_src:
            fsrc.write(k)
    
    with open(output_path + "train.tgt", 'w') as ftgt:
        for k in filtered_tgt:
            ftgt.write(k)


if __name__ == "__main__":
    main()