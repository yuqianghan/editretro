# Interpretable Retrosynthesis Prediction via Generative Editing Model 

The directory contains source code of the article: Interpretable Retrosynthesis Prediction via Generative Editing Model.

In this work, we propose an sequence edit-based retrosynthesis prediction method, called EditRetro, which formulaltes single-step retrosynthesis as a molecular string editing task. EditRetro offers an interpretable prediction process by performing explicit Levenshtein sequence editing operations, starting from the target product string. 
<div align=center>
<img src=model.jpg width="600px">
</div>

## Setup
Our code is based on facebook fairseq-0.9.0 version modified from https://github.com/weijia-xu/fairseq-editor and https://github.com/nedashokraneh/fairseq-editor.
Before installing fairseq, please place the clib files in EditRetro/clib into fairseq/clib files and EditRetro/fairseq_cli files moved into fairseq_cli files.

```
conda create -n editretro python=3.10.9
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
cd fairseq
python setup.py build develop
```

## Preprocess data
- The original datasets used in this paper are from:

   USPTO-50K: https://github.com/Hanjun-Dai/GLN

   USPTO-MIT: https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/data.zip

   USPTO-FULL: https://github.com/Hanjun-Dai/GLN

- Download raw datasets and put them in the ./data_process/raw_datasets folder, and then run the command:
```python
    python ./data_process/generate_aug_spe.py -dataset USPTO_50K -augmentation 20 -processes 8
    python ./data_process/generate_aug_spe.py -dataset USPTO-MIT -augmentation 5 -processes 8
    python ./data_process/generate_aug_spe.py -dataset USPTO_full -augmentation 5 -processes 8
```
- Then binarize the data using fairseq
```shell
    in_dir=data_after_aug_and_spe/uspto50k_aug10
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

## Edit with our prepared checkpoint
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

## Others
Should you have any questions, please contact Yuqiang Han at hyq2015@zju.edu.cn.
