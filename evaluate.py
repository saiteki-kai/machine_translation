import argparse
import typing

import numpy as np

from datasets import Dataset, load_dataset

from src.evaluation.metrics import comet_score, load_comet


def main(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = typing.cast(Dataset, dataset)

    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")

    for field in args.fields:
        data = [{"src": src, "mt": mt} for src, mt in zip(dataset[field], dataset[field + args.suffix], strict=True)]

        model = load_comet(args.comet_model)
        scores, _ = comet_score(model, data, batch_size=args.batch_size)

    print(f"{field} score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the translations using the COMET (reference-free) metric.")
    # Model arguments
    parser.add_argument("--comet-model", type=str, default="Unbabel/wmt22-cometkiwi-da", help="Comet model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, required=True, help="Huggingface dataset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split.")
    parser.add_argument("--fields", type=str, nargs="+", default=["prompt", "response"], help="Fields to evaluate.")
    parser.add_argument("--suffix", type=str, default="_it", help="Suffix for translated fields.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
