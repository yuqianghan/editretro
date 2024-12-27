- Download raw datasets and put them in the raw_datasets folder, and do SPE tokenization:

```
-  python preprocess_data.py -dataset USPTO_50K -augmentation 20 -processes 8 -spe -dropout 0 
-  python preprocess_data.py -dataset USPTO_FULL -augmentation 10 -processes 8 -spe -dropout 0
```

- Data processing using fairseq, for example, USPTO_50K,
```shell
sh binarize.sh ../datasets/USPTO_50K/aug20 dict.txt
```