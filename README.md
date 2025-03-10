# Machine Translation

## Installation

Clone the repository:

```bash
git clone https://github.com/saiteki-kai/machine_translation
```

Setup the environment and install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

Translate the dataset using the [X-ALMA](https://github.com/fe1ixxu/ALMA) model:

```bash
python translate.py \
    --model-name "haoranxu/X-ALMA-13B-Group2" \
    --max-length 400 512 \
    --dataset-name "PKU-Alignment/BeaverTails" \
    --split "330k_test" \
    --fields "prompt" "response" \
    --suffix "_it" \
    --num-beams 5 \
    --max-new-tokens 400 512 \
    --temperature 0.6 \
    --top-p 0.9 \
    --do-sample \
    --save-to-disk \
    --output-dir "data" \
    --push-to-hub \
    --repo-name "saiteki-kai/BeaverTails-it"
```

Evaluate the translations using the [COMET](https://github.com/Unbabel/COMET) model:

```bash
python evaluate.py \
    --comet-model "Unbabel/wmt22-cometkiwi-da" \
    --dataset-name "saiteki-kai/BeaverTails-it" \
    --split "330k_test" \
    --batch-size 32 \
    --fields "prompt" "response" \
    --suffix "_it"
```
