#!/bin/sh

python translate.py \
    --model-name "haoranxu/X-ALMA-13B-Group2" \
    --max-length 512 \
    --dataset-name "PKU-Alignment/BeaverTails" --split "330k_test" \
    --num-beams 1 --max-new-tokens 512 --temperature 0.0 --top-p 1.0 \ # greedy decoding
    --save-to-disk --output-dir "data" \
    --push-to-hub --repo-name "saiteki-kai/BeaverTails-it"

python evaluate.py \
    --dataset-name "saiteki-kai/BeaverTails-it" --split "330k_test" \
    --batch-size 32 \
    --comet-model "Unbabel/wmt22-cometkiwi-da" \

