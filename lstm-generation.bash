#!/bin/bash

./dataset/assert-no-duplicates.py --dataset ./dataset/raw --full-check

if [[ $1 -eq "t" || $1 -eq "-t" || $1 -eq "train" || $1 -eq "--train" ]]; then

    python3 main.py -v \
        --dataset ./dataset/raw --tokenize word --batch-size 2 \
        --sequence-length 50 \
        --lstm --lstm-embedding-dim 200 --lstm-hidden-dim 256 \
        --lstm-layers 2 --lstm-dropout 0.2 \
        --lstm-loss cross_entropy --lstm-optimizer adam --lstm-learning-rate 0.001 \
        --lstm-epochs 25 --lstm-save-model models/lstm-model.pt \
        --generate-recipes

elif [[ $1 -eq "g" || $1 -eq "-g" || $1 -eq "generate" || $1 -eq "--generate" ]]; then

    python3 main.py -v \
        --dataset ./dataset/raw --tokenize word \
        --lstm --lstm-load-model models/lstm-model.pt \
        --lstm-num-sentences 20 \
        --generate-recipes 1

fi

