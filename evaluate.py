import typing

import numpy as np

from datasets import Dataset, load_dataset

from src.evaluation.metrics import comet_score, load_comet


if __name__ == "__main__":
    model = load_comet("Unbabel/wmt22-cometkiwi-da")

    for split in ["330k_test", "330k_train"]:
        dataset = load_dataset("saiteki-kai/BeaverTails-it", split=split)
        dataset = typing.cast(Dataset, dataset)

        data = [
            {"src": src, "mt": mt}
            for src, mt in zip(
                dataset["prompt"] + dataset["response"],
                dataset["prompt_it"] + dataset["response_it"],
                strict=True,
            )
        ]

        scores, _ = comet_score(model, data, batch_size=32)

        print(split)
        print("-" * len(split))
        print(f"Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
