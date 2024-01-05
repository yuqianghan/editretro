#!/bin/bash

aug=10
topk=10
input='./datasets/USPTO_50K/aug/test.src'
name=uspto50k_best
databin='./datasets/USPTO_50K/bin'
root_dir='./results/finetune_50k/XXXXXX'
model_dir=${root_dir}/checkpoints
ckpt_name='checkpoint_epochbest.pt'
outputdir=$root_dir/generations
mkdir -p $outputdir

python ./utils/average_checkpoints.py --inputs $model_dir \
    --output $model_dir/$ckpt_name \
    --num-epoch-checkpoints 5 \
	# --num-update-checkpoints 5


CUDA_VISIBLE_DEVICES="0" CUDA_LAUNCH_BLOCKING=1 fairseq-interactive \
	--user-dir editretro \
	$databin \
	-s src -t tgt \
	--input $input \
	--task translation_retro \
	--path $model_dir/$ckpt_name \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 --remove-bpe \
    --init-src \
    --buffer-size 3000 \
    --batch-size 200 \
	--TOPK $topk \
	--print-step  --retain-iter-history >$outputdir/${name}.txt


python ./utils/extract.py \
	-generate_path $outputdir/${name}.txt \
	-outpath $outputdir/${name}.json \
	-predflag H \
	-tgt_path ./datasets/USPTO_50K/aug/test.tgt


python ./utils/score.py \
	-n_best $topk \
	-beam_size $topk \
	-predictions $outputdir/${name}.json \
	-targets $outputdir/${name}.json \
	-augmentation $aug \
	-score_alpha 0.1
