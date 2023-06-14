#!/bin/bash

TASK=${1}

function train_CAP() {
    python train_CAP.py \
        --task 'CAP' \
        --in_dim '128' \
        --out_dim '32' \
        --pool_dim '32' \
        --num_sect '4' \
        --rotation '45' \
        --head_sect '2' \
        --num_ring '3' \
        --bucket_interval '2.5,1.5' \
        --head_ring '2' \
        --drop '0.7' \
        --lr '1e-3' \
        --decay '0.01' \
        --epoch_num '2000' \
        --seed 42 \
        --cuda 1
}

function train_CP() {
    python train_CP.py \
        --task 'CP' \
        --in_dim '14' \
        --out_dim '32' \
        --pool_dim '128' \
        --num_sect '4' \
        --rotation '45' \
        --head_sect '2' \
        --num_ring '3' \
        --bucket_interval '2.5,1.5' \
        --head_ring '2' \
        --drop '0.5' \
        --lr '1e-3' \
        --decay '0.01' \
        --epoch_num '5000' \
        --seed 42 \
        --cuda 1
}

function train_DRSD() {
    python train_DRSD.py \
        --task 'DRSD' \
        --in_dim '64' \
        --out_dim '32' \
        --pool_dim '32' \
        --num_sect '4' \
        --rotation '45' \
        --head_sect '2' \
        --num_ring '2' \
        --bucket_interval '0.1,0.2' \
        --head_ring '2' \
        --drop '0.7' \
        --lr '1e-3' \
        --decay '0.0001' \
        --epoch_num '2000' \
        --seed 42 \
        --cuda 1
}

train_${TASK}