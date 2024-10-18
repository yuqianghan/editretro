# EditRetro: Retrosynthesis Prediction with an Iterative String Editing Model

The directory contains source code of the article: Retrosynthesis Prediction with an Iterative String Editing Model.

In this work, we propose an sequence edit-based retrosynthesis prediction method, called EditRetro, which formulaltes single-step retrosynthesis as a molecular string editing task. EditRetro offers an interpretable prediction process by performing explicit Levenshtein sequence editing operations, starting from the target product string.
<div align=center>
<img src=figures/workflow.png width="550px">
</div>


## Setup

- Create the environment:

```
conda create -n editretro python=3.10.9
pip install -r requirements.txt
```

&nbsp;&nbsp;&nbsp; You can install pytorch following the command:
```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

- Install fairseq to enable the use of our model:

```
git clone https://github.com/yuqianghan/editretro.git
cd  editretro/fairseq
pip install --editable ./
```

&nbsp;&nbsp;&nbsp; **Remarks !!!**: 
1. Set export CUDA_HOME=/usr/local/cuda in .bashrc;
2. Please verify the versions of CUDA (11.6.0) and gcc (9.4.0);
3. To ensure a successful installation of fairseq, please make sure to install Ninja first.
```
sudo apt install re2c
sudo apt-get install ninja-build
```
4. **If installed successfully !!!**, a file named _fairseq/fairseq/libnat_cuda.cpython-310-x86_64-linux-gnu.so should have been generated.


## Preprocess data
 The original datasets used in this paper are from:

   USPTO-50K: https://github.com/Hanjun-Dai/GLN  (schneider50k)

   <!-- USPTO-MIT: https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/data.zip -->

   USPTO-FULL: https://github.com/Hanjun-Dai/GLN  (1976_Sep2016_USPTOgrants_smiles.rsmi or uspto_multi)

> Remark: USPTO_FULL dataset. The raw version of USPTO is 1976_Sep2016_USPTOgrants_smiles.rsmi. The script for cleaning and de-duplication can be found under gln/data_process/clean_uspto.py. If you run the script on this raw rsmi file, you are expected to get the same data split as used in the GLN paper. Or you can download the cleaned USPTO dataset released by the authors (see uspto_multi folder under their dropbox folder).

Download **raw** datasets and put them in the _editretro/datasets/XXX(e.g., USPTO_50K)/raw_ folder, and then run the command to get the preprocessed datasets which will be stored in _editretro/datasets/XXX/aug_:

```python
cd preprocess
python preprocess_data.py -dataset USPTO_50K -augmentation 20 -processes 64 -spe
python preprocess_data.py -dataset USPTO_FULL -augmentation 10 -processes 64 -spe
```

Then binarize the data using 
```shell
sh binarize.sh ../datasets/USPTO_50K/aug20 dict.txt
```


## Traing the model
<!-- #### Pretrain and Finetune -->
To pretrain the model on the prepared dataset, please follow the instructions in **scripts/pretrain_readme.md**.

To finetune on specific dataset, please use **scripts/1_finetune_50k.sh** and **scripts/1_finetune_full.sh**
```
cd editretro (the root dicrectory)
sh ./scripts/1_finetune_50k.sh   or
sh ./scripts/1_finetune_full.sh
```


## Inference
To generate and score the predictions on the test set with binarized data:
```shell
sh  ./scripts/2_generate_50k.sh  or
sh  ./scripts/2_generate_full.sh
```

You can use your own trained checkpoint or use our prepared one available at https://drive.google.com/drive/folders/1em_I-PN-OvLXuCPfzWzRAUH-KZvSFL-U?usp=sharing. Refer to the 2_generate_xxx.sh script to specify the checkpoint.

Our method achieves the state-of-the-art performance on both USPTO-50K and USPTO-FULL datasets. 
<div align=center>
<img src=figures/results.png width="600px">
</div>

We achieve even better performance with the newly provided checkpoint on the USPTO-FULL dataset https://drive.google.com/file/d/19nb3Y6dro6DaEXu6MNRXui48LRE9C7tp/view?usp=drive_link

| Model         | Top-1    | Top-3    | Top-5    | Top-10   | 
| ------------- | -------- | -------- | -------- | -------- | 
| R-SMILES      | 48.9     | 66.6     | 72.0     | 76.4     |
| EditRetro     | 52.2     | 67.1     | 71.6     | 74.2     |
| EditRetro     | **54.3** | **67.6** | 71.1     | 74.3     | 



## Interactive Inference
After download the checkpoints on USPTO-50K and USPTO-FULL https://drive.google.com/drive/folders/1em_I-PN-OvLXuCPfzWzRAUH-KZvSFL-U?usp=sharing, you can iteractively edit your own molecule following the _**interactive/README**_
```shell
sh ./interactive/interactive_gen.sh
```




## Citation
```
@article{han2024editretro,
	title={Retrosynthesis Prediction with an Iterative String Editing Model},
	author={Han, Yuqiang et al.},
	journal={nature communications},
	year={2024}
}
```

## Reference
Our code is based on facebook fairseq-0.9.0 version modified from https://github.com/weijia-xu/fairseq-editor and https://github.com/nedashokraneh/fairseq-editor.

## Others
Should you have any questions, please contact Yuqiang Han at hyq2015@zju.edu.cn.
