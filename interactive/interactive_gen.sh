#!/bin/bash

gpus=0
aug=20    # the augmentations performed to the given product 
topk=20
beam_size=20  # beam_size = repos_beam * token_beam
repos_beam=5
token_beam=4

databin=interactive/databin
input_file=interactive/input_products.txt  # the given product SMILES
model_path=/path/to/the/model_checkpoint # 

output_dir=interactive
mkdir -p ${output_dir}

generation_file=${output_dir}/generation.txt
save_file=${output_dir}/ranked_preds.txt


CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-interactive \
	--user-dir editretro \
	${databin} \
	-s src -t tgt \
	--input ${input_file} \
	--task translation_retro \
	--path ${model_path} \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 --remove-bpe \
    --init-src \
    --buffer-size 3000 \
    --batch-size 200 \
	--TOPK ${topk} \
	--inference-with-augmentation \
	--aug ${aug} \
	--repos-beam ${repos_beam} \
	--mask-beam 1 \
	--token-beam ${token_beam} \
	--print-step --retain-iter-history > ${generation_file}


python ./utils/get_ranked_topk.py \
	-output_file  ${generation_file} \
	-save_file ${save_file} \
	-augmentation ${aug} \
	-beam_size ${beam_size} \
	-n_best ${topk} \
	-score_alpha 0.1 \
	# -output_edit_step

