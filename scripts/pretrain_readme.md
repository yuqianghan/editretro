
<div align=center>
<img src=../figures/pretrain.png width="550px">
</div>


## Prepare data
```
cd preprocess
python preprocess_data.py -dataset USPTO_50K -augmentation 20 -processes 64 -spe
python preprocess_data.py -dataset USPTO_FULL -augmentation 10 -processes 64 -spe -train_only -batch 50000    ### change the -batch according to the number of CPUs you have.
python preprocess_data.py -dataset USPTO_FULL -augmentation 5 -processes 64 -spe -train_except
```

## Filter data
> The filtered datasets will be stored in **datasets/USPTO_Pretrain/pretrain**.
```
python filter_pretrain.py
```
> Then binarize the datasets
```
bash binarize.sh ../datasets/USPTO_Pretrain/pretrain dict.txt
```

## Pretrain the model
```
bash ./scripts/0_pretrain.sh
```

> Remark: To fully leverage the diversity of the USPTO_FULL dataset, we employed a 10x augmentation of the training set to pretrain the model. To prevent any potential data leakage, we implemented the following solutions:
1. We filtered out the product molecules from the USPTO_50K test set.
2. We exclusively pretrained the Token Decoder out of the three (reposition, placeholder, and token decoders).
3. We fed the masked source sequence (mask ratio: 0.15) into the cross-attention mechanism.
