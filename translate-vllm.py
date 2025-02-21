import argparse
import typing

from pathlib import Path

from datasets import Dataset, load_dataset
from vllm import LLM, CompletionOutput, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


def post_process(output: CompletionOutput, model_name: str) -> str:
    if output.finish_reason == "length":
        print("Output truncated due to length.")

    text = output.text

    if "x-alma" in model_name.lower():
        if "[/INST]" in text:
            text = text.split("[/INST]")[1]

        return text.strip()

    if "Italian:" in text:
        text = text.split("Italian:")[2]

    return text.strip()


def apply_chat(input_text: str) -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "user",
            "content": f"Translate this from English into Italian:\nEnglish: {input_text}\nItalian:",
        },
    ]


def main(args: argparse.Namespace) -> None:
    model = LLM(args.model_name, dtype=args.dtype)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)  # greedy search

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = typing.cast(Dataset, dataset)

    translated_fields = {}

    for field in args.fields:
        if field not in dataset.features:
            print(f"Field '{field}' not found in the dataset. Skipping.")
            continue

        messages = [apply_chat(example) for example in dataset[field]]
        requests = model.chat(messages, sampling_params=sampling_params)

        translations = [post_process(req.outputs[0], args.model_name) for req in requests]
        translated_fields[field + args.suffix] = translations

    # add new fields to the dataset
    dataset = typing.cast(dict, dataset.to_dict())
    dataset.update(translated_fields)
    dataset = Dataset.from_dict(dataset)

    # save dataset
    data_path_dir = Path(args.output_dir, args.dataset_name.split("/")[-1], args.split)
    data_path_dir.mkdir(exist_ok=True, parents=True)
    dataset.save_to_disk(data_path_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate from English to Italian using a decoder-only LLM.")
    # Model arguments
    parser.add_argument("--model-name", type=str, default="haoranxu/X-ALMA-13B-Group2", help="Huggingface model.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, default="PKU-Alignment/BeaverTails", help="Huggingface dataset.")
    parser.add_argument("--split", type=str, default="30k_test", help="Dataset split.")
    parser.add_argument("--fields", type=str, nargs="+", default=["prompt", "response"], help="Fields to translate.")
    parser.add_argument("--suffix", type=str, default="_it", help="Suffix for translated fields.")
    # Saving arguments
    parser.add_argument("--output-dir", type=str, help="Directory where to save the dataset.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
