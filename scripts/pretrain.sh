#!/bin/bash

noise_type=random_delete
gpus="0,1,2,3"
model_args="--fp16"
architecture=pretrain_mlm_editretro
update=2
max_tokens=12000
max_epoch=500
exp_n=pretrain_full
run_n=$(date "+%Y%m%d_%H%M%S")

databin=./datasets/USPTO_FULL/bin # databin processed by fairseq

root_dir='./results'
exp_dir=$root_dir/$exp_n
mkdir -p $exp_dir


gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo $gpu_ids | wc -w)

echo "run_n:$run_n, max_tokens:$max_tokens, databin=${databin}, noise_type=${noise_type}, architecture=${architecture}, arguments=${model_args}, ${gpu_ids}, ${gpu_n}" > $exp_dir/config$run_n
cat $exp_dir/config$run_n

model_dir=${exp_dir}/$run_n/checkpoints
mkdir -p ${model_dir}


CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-train \
    $databin   \
    --user-dir editretro \
    -s src \
    -t tgt \
    --save-dir ${model_dir}  \
    --ddp-backend=no_c10d \
    --task translation_pretrain \
    --criterion pretrain_nat_loss \
    --arch ${architecture} \
    --noise ${noise_type} \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --update-freq ${update} \
    --max-tokens-valid 4000 \
    --distributed-world-size ${gpu_n}    \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens ${max_tokens} \
    --save-interval-updates 10000 \
    --max-update 500000  \
    --max-epoch ${max_epoch} \
    --keep-last-epochs 50 \
    --seed 1 \
    --mask-prob 0.15 \
    --pretrain \
    ${model_args} > ${model_dir}/pretrain.log
