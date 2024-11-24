import argparse
import json
import logging.config
import typing

from pathlib import Path

import numpy as np

from datasets import Dataset, load_dataset

from src.evaluation.metrics import comet_score, load_comet


logger = logging.getLogger("evaluation")


def setup_logging(config_path: Path):
    with config_path.open() as f:
        config = json.load(f)

    logging.config.dictConfig(config)


def main(args: argparse.Namespace) -> None:
    logger.debug("Arguments:")
    for k, v in vars(args).items():
        logger.debug(f"\t{k}: {v}")

    # Load the dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = typing.cast(Dataset, dataset)
    logger.info(f"Dataset '{args.dataset_name}' (split: {args.split}) loaded.")
    logger.debug(f"Dataset size: {len(dataset)}")

    # Load the model
    model = load_comet(args.comet_model)
    logger.info(f"Model '{args.comet_model}' loaded.")

    total_scores = []

    for field in args.fields:
        # Check if the fields are present in the dataset
        if field not in dataset.column_names or (field + args.suffix) not in dataset.column_names:
            logger.error(f"Field '{field}' and '{field + args.suffix}' must be present in the dataset.")
            continue

        # Prepare data for the model input
        data = [{"src": src, "mt": mt} for src, mt in zip(dataset[field], dataset[field + args.suffix], strict=True)]

        # Evaluate the translations using COMET (reference-free metric)
        logger.info(f"Starting evaluation for {field}")
        scores, _ = comet_score(model, data, batch_size=args.batch_size)
        total_scores.extend(scores)

        logger.info(f"{field.capitalize()} score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    logger.info(f"Total score: {np.mean(total_scores):.2f} ± {np.std(total_scores):.2f}")


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
    setup_logging(config_path=Path("configs") / "logging.json")
    main(args)
