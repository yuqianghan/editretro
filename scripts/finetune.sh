#!/bin/bash

run_n=$(date "+%Y%m%d_%H%M%S")
exp_n=finetune_50k

root_dir=./results
exp_dir=$root_dir/$exp_n
mkdir -p $exp_dir

model_dir=${exp_dir}/$run_n/checkpoints
mkdir -p $model_dir

databin=./datasets/USPTO_50K/bin
pretrain_ckpt_path=$root_dir/pretrain_full/xxxx/checkpoints # pretrain checkpoint path

python ./utils/average_checkpoints.py --inputs $pretrain_ckpt_path \
    --output $pretrain_ckpt_path/avg.pt \
    --num-epoch-checkpoints 20

python ./utils/pretrain_ckpt_utils.py \
	--inputckpt $pretrain_ckpt_path/avg.pt \
	--outputckpt $pretrain_ckpt_path/avg.pt

pretrain_ckpt=$pretrain_ckpt_path/avg.pt

gpus="0"
gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo $gpu_ids | wc -w)

CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-train \
    --user-dir editretro \
	$databin \
	--save-dir $model_dir \
	--ddp-backend=no_c10d \
	--task translation_retro \
	--criterion nat_loss \
	--arch editretro_nat \
	--noise random_delete_shuffle \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr 0.0001 --lr-scheduler inverse_sqrt \
	--min-lr '1e-09' --warmup-updates 10000 \
	--warmup-init-lr '1e-07' --label-smoothing 0.1 \
	--share-all-embeddings \
	--dropout 0.3 --weight-decay 0.001 \
	--decoder-learned-pos --encoder-learned-pos \
	--apply-bert-init \
	--max-tokens-valid 4000 \
	--log-format 'simple' \
	--log-interval 100 \
	--fixed-validation-seed 7 \
	--max-tokens 8000 \
	--save-interval-updates 10000 \
	--keep-last-epochs 20 \
	--max-epoch 50 \
	--max-update 300000 \
	--alpha-ratio 0.2 \
	--dae-ratio 0.2 \
	--fp16  \
	--restore-file $pretrain_ckpt \
	--reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader \
	--distributed-world-size $gpu_n > ${model_dir}/finetune.log 
