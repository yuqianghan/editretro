run_n=1
exp_n=trainfromscratch

root_dir=`dirname $0`
exp_dir=$root_dir/exp_$exp_n
mkdir -p $exp_dir

model_dir=${exp_dir}/checkpoints$run_n
mkdir -p $model_dir

databin=data-bin # databin processed by fairseq
pretrain_ckpt=""

gpus="1"
gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo $gpu_ids | wc -w)

CUDA_VISIBLE_DEVICES=$gpus fairseq-train \
    --user-dir EditRetro \
	$databin \
	--save-dir $model_dir \
	--ddp-backend=no_c10d \
	--task translation_retro \
	--criterion nat_loss \
	--arch editretro \
	--noise random_delete_shuffle \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr 0.0002 --lr-scheduler inverse_sqrt \
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
	--max-tokens 15000 \
	--distributed-world-size $gpu_n \
	--save-interval-updates 10000 \
	--keep-last-epochs 50 \
	--max-epoch 200 \
	--max-update 300000 \
	--alpha-ratio 0.2 \
	--dae-ratio 0.2 \
	--fp16  \
	--distributed-world-size $gpu_n > ${model_dir}/log 
