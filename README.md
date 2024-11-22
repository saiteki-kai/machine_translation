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
    --max-length 512 \
    --dataset-name "PKU-Alignment/BeaverTails" \
    --split "330k_test" \
    --num-beams 5 \
    --max-new-tokens 512 \
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
    --batch-size 32
```

## TODO

- [x] compute dataset statistics (e.g. max length, average length, total number of tokens)
- [x] compute translation quality metrics (reference-free)
- [x] compute translation quality heuristics (e.g. length ratio, question preserved)
- [x] save translations
- [ ] define which hyperparameters to use (beam size, temperature, top-p, max_new_tokens)
- [ ] create the italian version of the dataset
- [ ] adapt the translate.py script for the beavertails-evaluation dataset
- [ ] translate other benchmark datasets
