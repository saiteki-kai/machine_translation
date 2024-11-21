import argparse
import typing

from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import GenerationConfig

from src.translation.translator import ALMATranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate using the ALMA model.")
    # Model arguments
    parser.add_argument("--model-name", type=str, default="haoranxu/X-ALMA-13B-Group2", help="Huggingface model.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum length for tokenized input.")
    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, default="PKU-Alignment/BeaverTails", help="Huggingface dataset.")
    parser.add_argument("--split", type=str, default="30k_test", help="Dataset split.")
    # Generation arguments
    parser.add_argument("--num-beams", type=int, default=5, help="Number of beams for beam search.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    # Saving arguments
    parser.add_argument("--save-to-disk", action="store_true", help="Save dataset to disk.")
    parser.add_argument("--output-dir", type=str, help="Directory where to save the dataset.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to Huggingface Hub.")
    parser.add_argument("--repo-name", type=str, help="Huggingface repo where to push the dataset.")

    args = parser.parse_args()

    if args.push_to_hub and args.repo_name is None:
        parser.error("--push-to-hub requires --repo-name")

    if args.save_to_disk and args.output_dir is None:
        parser.error("--save-to-disk requires --output-dir")

    return args


if __name__ == "__main__":
    args = parse_args()

    model = ALMATranslator(args.model_name)

    gen_config = GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    def en_to_it_prompt(text: str) -> str:
        return f"Translate this from English to Italian:\nEnglish: {text}\nItalian:"

    def translate(example: dict[str, str]) -> dict[str, str]:
        prompt_it = model.translate(
            example["prompt"],
            max_length=args.max_length,
            generation_config=gen_config,
            translation_prompt=en_to_it_prompt,
        )

        response_it = model.translate(
            example["response"],
            max_length=args.max_length,
            generation_config=gen_config,
            translation_prompt=en_to_it_prompt,
        )

        return {"prompt_it": prompt_it, "response_it": response_it}

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = typing.cast(Dataset, dataset.map(translate))

    if args.save_to_disk:
        data_path_dir = Path(args.output_dir, args.dataset_name.split("/")[-1], args.split)
        data_path_dir.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(data_path_dir)

    if args.push_to_hub:
        dataset.push_to_hub(args.repo_name, split=args.split)
