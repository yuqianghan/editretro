databin=''

run_n=1
exp_n=finetune

root_dir=`dirname $0`
exp_dir=$root_dir/exp_$exp_n
model_dir=${exp_dir}/checkpoints$run_n
ckpt_name=avg.pt

aug=10
topk=10

name=single
input=inputs_example.txt # src

outputdir=$model_dir/output
mkdir -p $outputdir

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
	--inference-with-augmentation \
	--aug $aug \
	--print-step --retain-iter-history >$outputdir/${name}.txt

python EditRetro/utils/get_ranked_topk.py \
	-output_file $outputdir/${name}.txt \
	-save_file $outputdir/ranked_output.txt \
	-augmentation $aug \
	-beam_size $topk \
	-n_best $topk \
	-score_alpha 0.1 \
	-output_edit_step

