#!/bin/bash

gpus='0'
aug=5
topk=10
beam_size=10
repos_beam=5
token_beam=2
max_tokens=2500  #!!!TODO: reduce this number if encountering CUDA OOM

databin=./datasets/USPTO_FULL/aug10/data-bin  # binaried test data, the test.src are 5x augmented
root_dir=results/finetune_50k/xxxxxxxx_xxxxxx  #TODO: point to the pretrain checkpoint path
model_dir=${root_dir}/checkpoints

exp_n=1
outfile=generation
outputdir=${root_dir}/generations/$exp_n
mkdir -p $outputdir

ckpt_name=finetune.pt
ckpt_path=${outputdir}/${ckpt_name}

python ./utils/average_checkpoints.py --inputs ${model_dir} \
    --output ${ckpt_path} \
    --num-epoch-checkpoints 5 \
	# --checkpoint-upper-bound 500 \
    # --num-update-checkpoints 5 \

CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-generate \
	--user-dir editretro \
	$databin \
	-s src -t tgt \
	--gen-subset test \
	--task translation_retro \
	--path ${ckpt_path} \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 --remove-bpe \
	--init-src \
	--TOPK ${beam_size} \
	--max-tokens ${max_tokens} \
	--repos-beam ${repos_beam} \
	--mask-beam 1 \
	--token-beam ${token_beam} \
	--fp16 \
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
