root_dir=./datasets/USPTO_FULL
input_dir=aug_5
out_dir=bin_5
dict=./preprocess/dict.txt
src=src
tgt=tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $root_dir/$input_dir/train --validpref $root_dir/$input_dir/val --testpref $root_dir/$input_dir/test \
    --destdir $root_dir/$out_dir \
    --workers 20 \
    --tgtdict $dict \
    --srcdict $dict