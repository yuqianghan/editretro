
-Download raw datasets and put them in the raw_datasets folder, and then run the command:

```python
python generate_PtoR_data.py -dataset USPTO_50K -augmentation 20 -processes 8
python generate_PtoR_data.py -dataset USPTO-MIT -augmentation 5 -processes 8
python generate_PtoR_data.py -dataset USPTO_full -augmentation 5 -processes 8
```


-Preprocess data using fairseq
```shell
in_dir=data_after_aug_and_spe/uspto50k_aug10
out_dir=data_bin

databin_name=uspto50k_aug10
src=src
tgt=tgt
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $in_dir/train --validpref $in_dir/val --testpref $in_dir/test \
    --destdir $out_dir/data-bin_$databin_name \
    --workers 20 \
    --tgtdict dict.txt \
    --srcdict dict.txt
```