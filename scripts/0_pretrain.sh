#!/bin/bash

gpus='0'    # if you have multiple GPUs, you can use them by setting "0,1,2,3"
lr=0.0003   
max_tokens=49152   #  8192 x 6,  if you have access to 6 GPUs, set this to 8192
warmup=10000   
update=1    # if encounter CUDA OOM, you can use this parameter to do gradient accumulation 
max_epoch=300
max_update=300000   # running more updates to well pretrain the model
noise_type=random_mask
architecture=pretrain_mlm_editretro
task=translation_pretrain
criterion=pretrain_nat_loss

databin=datasets/USPTO_Pretrain/pretrain/data-bin   # point to the preprocessed data

exp_n=pretrain
root_dir=./results
run_n=$(date "+%Y%m%d_%H%M%S")
exp_dir=${root_dir}/$exp_n
mkdir -p ${exp_dir}

gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo ${gpu_ids} | wc -w)

model_dir=${exp_dir}/${run_n}/checkpoints
mkdir -p ${model_dir}

echo "run_n:${run_n}, max_tokens:${max_tokens}, databin=${databin}, noise_type=${noise_type}, architecture=${architecture}, ${gpu_ids}, ${gpu_n}" > ${exp_dir}/${run_n}/config.log
cat ${exp_dir}/${run_n}/config.log


CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-train \
    $databin   \
    --user-dir editretro \
    -s src \
    -t tgt \
    --save-dir ${model_dir}  \
    --ddp-backend=no_c10d \
    --task ${task}  \
    --criterion ${criterion} \
    --arch ${architecture} \
    --noise ${noise_type} \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr $lr --lr-scheduler inverse_sqrt \
    --clip-norm 0.0  \
    --warmup-updates ${warmup} \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --share-all-embeddings \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --max-tokens-valid 4000 \
    --log-format 'simple' --log-interval 300 \
    --fixed-validation-seed 7 \
    --max-tokens ${max_tokens} \
    --save-interval-updates 10000 \
    --max-update ${max_update}  \
    --max-epoch ${max_epoch} \
    --keep-last-epochs 20 \
    --seed 1 \
    --mask-prob 0.15 \
    --pretrain \
    --update-freq ${update} \
    --distributed-world-size ${gpu_n}    \
    ${model_args} > ${model_dir}/pretrain.log
