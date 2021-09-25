#!/bin/bash

./dataset/assert-no-duplicates.py --dataset ./dataset/raw --full-check

python3 main.py \
    --dataset ./dataset/raw --tokenize none \
    --markov --markov-state-size 3 --markov-num-sentences 20 \
    --generate-recipes
