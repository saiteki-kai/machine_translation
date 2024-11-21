import argparse
import typing

import numpy as np

from datasets import Dataset, load_dataset

from src.evaluation.metrics import comet_score, load_comet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the translations using the COMET (reference-free) metric.")
    # Model arguments
    parser.add_argument("--comet-model", type=str, default="Unbabel/wmt22-cometkiwi-da", help="Comet model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, required=True, help="Huggingface dataset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = typing.cast(Dataset, dataset)

    data = [
        {"src": src, "mt": mt}
        for src, mt in zip(
            dataset["prompt"] + dataset["response"],
            dataset["prompt_it"] + dataset["response_it"],
            strict=True,
        )
    ]

    model = load_comet(args.comet_model)
    scores, _ = comet_score(model, data, batch_size=args.batch_size)

    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
