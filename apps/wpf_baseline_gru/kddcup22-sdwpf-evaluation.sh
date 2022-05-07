#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    echo "The machine learning framework (i.e. base/paddlepaddle/pytorch/tensorflow) is missing, which MUST be provided"
    echo "For example, sh run_evaluation.sh paddlepaddle"
    exit 1
fi

ML_framework=$1
data_path="path/to/data"
filename="sdwpf_baidukddcup2022_full.csv"
path_to_test_x="path/to/data/sdwpf_baidukddcup2022_test/test_x"
path_to_test_y="path/to/data/sdwpf_baidukddcup2022_test/test_y"
predict_file="predict.py"
is_debug=True

conda activate $1
python "./kddcup22-sdwpf-evaluation/"$ML_framework"/evaluation.py" \
    --data_path $data_path \
    --filename $filename \
    --path_to_test_x $path_to_test_x \
    --path_to_test_y $path_to_test_y \
    --pred_file "./kddcup22-sdwpf-evaluation/"$ML_framework"/"$predict_file \
    --is_debug $is_debug \
    --framework $ML_framework
