
databin=''

run_n=1
exp_n=finetune

root_dir=`dirname $0`
exp_dir=$root_dir/exp_$exp_n
model_dir=${exp_dir}/checkpoints$run_n
ckpt_name=avg.pt
python EditRetro/utils/average_checkpoints.py --inputs $model_dir \
    --output $model_dir/$ckpt_name \
    --num-epoch-checkpoints 20


outputdir=$model_dir/output
mkdir -p $outputdir

aug=10
topk=10

input=input.aug$aug # src <sep> tgt 
name=output_name

CUDA_VISIBLE_DEVICES="1" CUDA_LAUNCH_BLOCKING=1 fairseq-interactive \
	--user-dir EditRetro \
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

python EditRetro/utils/extract.py \
	-generate_path $outputdir/${name}.txt \
	-outpath $outputdir/${name}.json \
	-predflag H \
	-tgt_path ./test.tgt

python EditRetro/utils/score.py \
	-n_best $topk \
	-beam_size $topk \
	-predictions $outputdir/${name}.json \
	-targets $outputdir/${name}.json \
	-augmentation $aug \
	-score_alpha 0.1