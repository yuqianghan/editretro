#!/bin/bash

gpus=$1
aug=20
topk=20
beam_size=20
outfile=generation

databin=./datasets/USPTO_50K/aug20/data-bin  # binaried test data
root_dir=results/finetune # the path to  the generation results
model_dir=${root_dir}/checkpoints
outputdir=${root_dir}/generations
mkdir -p $outputdir

ckpt_name=checkpoint_finetune.pt
ckpt_path=${model_dir}/${ckpt_name}

python ./utils/average_checkpoints.py --inputs ${model_dir} \
    --output ${ckpt_path} \
    --num-update-checkpoints 5  \
    # --num-epoch-checkpoints 10 \


CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-generate \
	--user-dir editretro \
	$databin \
	-s src -t tgt \
	--gen-subset test \
	--task translation_retro \
	--path ${ckpt_path} \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 1 \
	--beam 1 --remove-bpe \
	--init-src \
	--TOPK ${beam_size} \
	--max-tokens 50000 \
	--repos-beam 5 \
	--mask-beam 1 \
	--token-beam 4 \
	--print-step --retain-iter-history >$outputdir/${outfile}.txt \


# post processing
src=src.txt
tgt=tgt.txt
pred=pred.txt
prob=prob.txt
grep ^S ${outputdir}/${outfile}.txt | LC_ALL=C sort -V | cut -f2- > ${outputdir}/${src}
grep ^T ${outputdir}/${outfile}.txt | LC_ALL=C sort -V | cut -f2- > ${outputdir}/${tgt}
grep ^H ${outputdir}/${outfile}.txt | LC_ALL=C sort -V | cut -f3- > ${outputdir}/${pred}
grep ^P ${outputdir}/${outfile}.txt | LC_ALL=C sort -V | cut -f2- > ${outputdir}/${prob}


python ./utils/post_process.py \
    -generate_path  ${outputdir}/${pred} \
    -prob_path ${outputdir}/${prob} \
    -tgt_path ${outputdir}/${tgt} \
    -out_path ${outputdir}/${outfile}.json


# evaluate the results
python ./utils/score.py \
	-n_best ${topk} \
	-beam_size ${beam_size} \
	-predictions ${outputdir}/${outfile}.json \
	-targets ${outputdir}/${outfile}.json \
	-augmentation ${aug} \
	-score_alpha 0.1 \
