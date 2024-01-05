#!/bin/bash

aug=10
topk=10
epochs=5
upper=50
name=uspto50k_epoch${upper}
ckpt_name=checkpoint_epoch${upper}.pt
databin='./datasets/USPTO_50K/bin'
root_dir='./results/finetune_50k/XXXXXX'
model_dir=${root_dir}/checkpoints
outputdir=${root_dir}/generations
mkdir -p $outputdir

python ./utils/average_checkpoints.py --inputs $model_dir \
    --output $model_dir/$ckpt_name \
    --num-epoch-checkpoints ${epochs} \
	--checkpoint-upper-bound ${upper}
	# --num-update-checkpoints 5


CUDA_VISIBLE_DEVICES="2" CUDA_LAUNCH_BLOCKING=1 fairseq-generate \
	--user-dir editretro \
	$databin \
	-s src -t tgt \
	--gen-subset test \
	--task translation_retro \
	--path $model_dir/$ckpt_name \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 --remove-bpe \
	--init-src \
	--TOPK ${topk} \
	--max-tokens 4000 \
	--num-workers 4 \
	--print-step --retain-iter-history >$outputdir/${name}.txt


# post processing
src=src.txt
tgt=tgt.txt
pred=pred.txt
prob=prob.txt
grep ^S ${outputdir}/${name}.txt | LC_ALL=C sort -V | cut -f2- > ${outputdir}/${src}
grep ^T ${outputdir}/${name}.txt | LC_ALL=C sort -V | cut -f2- > ${outputdir}/${tgt}
grep ^H ${outputdir}/${name}.txt | LC_ALL=C sort -V | cut -f3- > ${outputdir}/${pred}
grep ^P ${outputdir}/${name}.txt | LC_ALL=C sort -V | cut -f2- > ${outputdir}/${prob}


python ./utils/post_process.py \
    -generate_path  ${outputdir}/${pred} \
    -prob_path ${outputdir}/${prob} \
    -tgt_path ${outputdir}/${tgt} \
    -out_path ${outputdir}/${name}.json


# evaluate the results
python ./utils/score.py \
	-n_best $topk \
	-beam_size $topk \
	-predictions $outputdir/${name}.json \
	-targets $outputdir/${name}.json \
	-augmentation $aug \
	-score_alpha 0.1


