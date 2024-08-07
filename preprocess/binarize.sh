input_dir=$1   # ../datasets/USPTO_50K/aug20
out_dir=${input_dir}/data-bin
dict=$2  # dict.txt in current folder
src=src
tgt=tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $input_dir/train --validpref $input_dir/val --testpref $input_dir/test \
    --destdir $out_dir \
    --workers 40 \
    --tgtdict $dict \
    --srcdict $dict \
    # --joined-dictionary \