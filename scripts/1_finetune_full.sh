#!/bin/bash

gpus='0'
lr=0.0003
max_tokens=40960
warmup=10000
update=1
max_epoch=2000
max_update=2000000
noise_type=random_delete_shuffle
architecture=editretro_nat
task=translation_retro
loss=nat_loss

exp_n=finetune_full
run_n=$(date "+%Y%m%d_%H%M%S")

root_dir=results
exp_dir=${root_dir}/${exp_n}
mkdir -p ${exp_dir}

model_dir=${exp_dir}/${run_n}/checkpoints
mkdir -p ${model_dir}

databin=datasets/USPTO_FULL/aug10/data-bin

pretrain_ckpt_path=results/pretrain/xxxxxxxx_xxxxxx/checkpoints  #TODO: point to the pretrain checkpoint path
ckpt_name=${pretrain_ckpt_path}/pretrain.pt

python utils/average_checkpoints.py --inputs ${pretrain_ckpt_path} \
    --output ${ckpt_name} \
	--num-update-checkpoints 5 \
    # --num-epoch-checkpoints 10 \
	# --checkpoint-upper-bound 10 \

python utils/pretrain_ckpt_utils.py \
	--inputckpt ${ckpt_name} \
	--outputckpt ${ckpt_name}

gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo ${gpu_ids} | wc -w)

CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-train \
	$databin \
    --user-dir editretro \
	-s src \
	-t tgt \
	--save-dir ${model_dir} \
	--ddp-backend=no_c10d \
	--task ${task} \
	--criterion ${loss} \
	--arch ${architecture} \
	--noise ${noise_type} \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr ${lr} --lr-scheduler inverse_sqrt \
	--min-lr '1e-09' --warmup-updates ${warmup} \
	--warmup-init-lr '1e-07' \
	--label-smoothing 0.1 \
	--dropout 0.1 --attention-dropout 0.1 \
	--weight-decay 0.01 \
	--share-all-embeddings \
	--decoder-learned-pos --encoder-learned-pos \
	--max-tokens-valid 4000 \
	--log-format 'simple' \
	--log-interval 500 \
	--fixed-validation-seed 7 \
	--max-tokens ${max_tokens} \
	--save-interval-updates 10000 \
	--keep-last-epochs 20 \
	--max-epoch ${max_epoch} \
	--max-update ${max_update} \
	--alpha-ratio 0.5 \
	--dae-ratio 0.5 \
	--fp16  \
	--update-freq ${update} \
	--restore-file ${ckpt_name} \
	--reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader \
	--distributed-world-size ${gpu_n} > ${model_dir}/finetune_full.log
