#!/bin/bash

gpus=$1
noise_type=random_delete_shuffle
model_args="--fp16"
architecture=editretro_nat
task=translation_retro
loss=nat_loss
update=1
lr=0.0001
max_tokens=8192
max_epoch=1500
max_update=300000

exp_n=finetune
run_n=$(date "+%Y%m%d_%H%M%S")

root_dir=results
exp_dir=$root_dir/$exp_n
mkdir -p $exp_dir

model_dir=${exp_dir}/$run_n/checkpoints
mkdir -p $model_dir


databin=./datasets/USPTO_50K/aug20/data-bin # databin processed by fairseq
pretrain_ckpt_path=results/pretrain/xxxxxx/checkpoints  # pretrain checkpoint path
pretrain_ckpt_name=${pretrain_ckpt_path}/checkpoint_pretrain.pt  # the name used to represent the pretrained ckpt


python utils/average_checkpoints.py --inputs $pretrain_ckpt_path \
    --output ${pretrain_ckpt_name} \
	--num-update-checkpoints 5 \
    # --num-epoch-checkpoints 10 \

python utils/pretrain_ckpt_utils.py \
	--inputckpt ${pretrain_ckpt_name} \
	--outputckpt ${pretrain_ckpt_name}


gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo $gpu_ids | wc -w)


CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-train \
    --user-dir editretro \
	$databin \
	--save-dir $model_dir \
	--ddp-backend=no_c10d \
	--task  $task  \
	--criterion $loss  \
	--arch $architecture  \
	--noise $noise_type  \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr $lr --lr-scheduler inverse_sqrt \
	--min-lr '1e-09' --warmup-updates 10000 \
	--warmup-init-lr '1e-07' --label-smoothing 0.1 \
	--share-all-embeddings \
	--dropout 0.3 --weight-decay 0.001 \
	--decoder-learned-pos --encoder-learned-pos \
	--max-tokens-valid 4000 \
	--log-format 'simple' \
	--log-interval 100 \
	--fixed-validation-seed 7 \
	--max-tokens 8000 \
	--save-interval-updates 10000 \
	--keep-last-epochs 20 \
	--max-epoch ${max_epoch} \
	--max-update ${max_update} \
	--alpha-ratio 0.2 \
	--dae-ratio 0.2 \
	--fp16  \
	--restore-file ${pretrain_ckpt_name} \
	--reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader \
	--distributed-world-size $gpu_n > ${model_dir}/finetune.log 
