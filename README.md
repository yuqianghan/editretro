# EditRetro: Advancing Retrosynthesis Prediction with Iterative Editing Model

The directory contains source code of the article: Advancing Retrosynthesis Prediction with Iterative Editing Model.

In this work, we propose an sequence edit-based retrosynthesis prediction method, called EditRetro, which formulaltes single-step retrosynthesis as a molecular string editing task. EditRetro offers an interpretable prediction process by performing explicit Levenshtein sequence editing operations, starting from the target product string. 
<div align=center>
<img src=model.jpg width="600px">
</div>

## Setup

- Create the environment:

 ```
conda create -n editretro python=3.10.9
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
 ```

- Download the fairseq-0.9.0 from https://github.com/facebookresearch/fairseq/releases/tag/v0.9.0 and unzip the file as fairseq-0.9.0.

- Install fairseq to enable it can be used by command line.
  
- Before installing fairseq, please place the clib files in _editretro/clib_ into _fairseq-0.9.0/fairseq/clib_ files and move _editretro/fairseq_cli_ files to _fairseq-0.9.0/fairseq_cli_.

```
git clone https://github.com/yuqianghan/editretro.git
cd  editretro
pip install --editable ./
python setup.py build_ext --inplace
```

```
# sudo apt install re2c
# sudo apt-get install ninja-build
```
Remarks: set export CUDA_HOME=/usr/local/cuda in .bashrc

## Preprocess data
- The original datasets used in this paper are from:

   USPTO-50K: https://github.com/Hanjun-Dai/GLN  (schneider50k)

   USPTO-MIT: https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/data.zip

   USPTO-FULL: https://github.com/Hanjun-Dai/GLN  (1976_Sep2016_USPTOgrants_smiles.rsmi or uspto_multi)

> Remark: USPTO_FULL dataset. The raw version of USPTO is 1976_Sep2016_USPTOgrants_smiles.rsmi. The script for cleaning and de-duplication can be found under gln/data_process/clean_uspto.py. If you run the script on this raw rsmi file, you are expected to get the same data split as used in the GLN paper. Or you can download the cleaned USPTO dataset released by the authors (see uspto_multi folder under their dropbox folder).

- Download **raw** datasets and put them in the _editretro/retro_epxs/data_process/raw_datasets_ folder, and then run the command to get the preprocessed datasets which will be stored in _processed_datasets_:
```python
    cd editretro/repro_exps/data_process
    python generate_aug_spe.py -dataset USPTO_50K -augmentation 10 -processes 8
    python generate_aug_spe.py -dataset USPTO-MIT -augmentation 5 -processes 8
    python generate_aug_spe.py -dataset USPTO_FULL -augmentation 5 -processes 8
```

- Then binarize the data using fairseq, for example, USPTO_50K_aug10:
```shell
    in_dir=./processed_datasets/USPTO_50K_aug10
    out_dir=data_bin
    databin_name=uspto50k_aug10
    src=src
    tgt=tgt
    fairseq-preprocess --source-lang $src --target-lang $tgt \
        --trainpref $in_dir/train \
        --validpref $in_dir/val \
        --testpref    $in_dir/test \
        --destdir $out_dir/data-bin_$databin_name \
        --workers 20 \
        --tgtdict dict.txt \
        --srcdict dict.txt
```

## Pretrain and Finetune
- Pre-train on the augmented USPTO-full datasets with jointly masked modeling:
```shell
    sh EditRetro/scripts/pretrain.sh
```
- Fine-tune on specific dataset, for example, USPTO-50K:
```shell
    sh EditRetro/scripts/fine-tune.sh
```
- The model can also be trained from scratch:
```shell
    sh EditRetro/scripts/train_from_scratch.sh
```


## Inference
To generate and score the predictions on the test set with mini-batch:
```shell
sh  EditRetro/scripts/inference.sh
```
Our method achieves the state-of-the-art performance on the USPTO-50K dataset. 
<div align=center>
<img src=results.png width="600px">
</div>

## Inference with our prepared checkpoint
After download the checkpoint trained on USPTO-50K https://drive.google.com/drive/folders/1em_I-PN-OvLXuCPfzWzRAUH-KZvSFL-U?usp=sharing, you can edit your own molecule:
```shell
sh EditRetro/scripts/generate_single.sh
```


<!-- 
## Citation
```
@article{han2023editretro,
	title={Explainable and Diverse Retrosynthesis Prediction via Generative Editing Model},
	author={Han, Yuqiang et al.},
	journal={},
	year={2023}
}
``` -->

## Reference
Our code is based on facebook fairseq-0.9.0 version modified from https://github.com/weijia-xu/fairseq-editor and https://github.com/nedashokraneh/fairseq-editor.

## Others
Should you have any questions, please contact Yuqiang Han at hyq2015@zju.edu.cn.
