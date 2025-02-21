python translate-vllm.py \
    --model-name "haoranxu/X-ALMA-13B-Group2" --dtype float16 \
    --max-new-tokens 512 \
    --dataset-name "PKU-Alignment/BeaverTails" --split "330k_test" \
    --fields "prompt" "response" --suffix "_it" \
    --output-dir "data"
