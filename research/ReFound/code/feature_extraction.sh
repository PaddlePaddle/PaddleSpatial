#!/bin/bash

shparam1=${1}

function region_feature_extraction() {
    CUDA_VISIBLE_DEVICES=0 \
    python -u feature_extractor.py \
    --city ${1} \
    --checkpoint '299' \
    --extract_batch_size '128' \
    --seed '42'
}

region_feature_extraction ${shparam1}