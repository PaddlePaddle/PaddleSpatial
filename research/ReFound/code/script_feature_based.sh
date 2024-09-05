#!/bin/bash

shparam1=${1}
shparam2=${2}
shparam3=${3}
shparam4=${4}
shparam5=${5}
shparam6=${6}
shparam7=${7}
shparam8=${8}


function train_uvd() {
    CUDA_VISIBLE_DEVICES=${7} \
    python -u feature_based_uvd.py \
    --city ${1} \
    --checkpoint '299' \
    --agg ${2} \
    --fdrop ${3} \
    --fbatch_size '12' \
    --epoch_num '40' \
    --warmup_epochs '3' \
    --flr ${4} \
    --fdecay ${5} \
    --min_lr '0' \
    --accum_iter '1' \
    --seed ${6}
}


function train_cap() {
    CUDA_VISIBLE_DEVICES=${7} \
    python -u feature_based_cap.py \
    --city ${1} \
    --checkpoint '299' \
    --agg ${2} \
    --fdrop ${3} \
    --fbatch_size '12' \
    --epoch_num '40' \
    --warmup_epochs '3' \
    --flr ${4} \
    --fdecay ${5} \
    --min_lr '0' \
    --accum_iter '1' \
    --seed ${6}
}


function train_pop() {
    CUDA_VISIBLE_DEVICES=${7} \
    python -u feature_based_pop.py \
    --city ${1} \
    --checkpoint '299' \
    --agg ${2} \
    --fdrop ${3} \
    --fbatch_size '12' \
    --epoch_num '40' \
    --warmup_epochs '3' \
    --flr ${4} \
    --fdecay ${5} \
    --min_lr '0' \
    --accum_iter '1' \
    --seed ${6}
}


train_${shparam1} ${shparam2} ${shparam3} ${shparam4} ${shparam5} ${shparam6} ${shparam7} ${shparam8}
