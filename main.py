import argparse

from pathlib import Path

from datasets import load_dataset
from transformers import GenerationConfig

from translator import ALMATranslator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate using the ALMA model.")
    # Model arguments
    parser.add_argument("--model-name", type=str, default="haoranxu/X-ALMA-13B-Group2", help="Huggingface model.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum length for tokenized input.")
    # Generation arguments
    parser.add_argument("--num-beams", type=int, default=5, help="Number of beams for beam search.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    # Saving arguments
    parser.add_argument("--save-to-disk", action="store_true", help="Save dataset to disk.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to Huggingface Hub.")

    args = parser.parse_args()

    model = ALMATranslator(args.model_name)

    gen_config = GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    def en_to_it_prompt(text) -> str:
        return f"Translate this from English to Italian:\nEnglish: {text}\nItalian:"

    def translate(example):
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

    for split in ["330k_test", "330k_train"]:
        dataset = load_dataset("PKU-Alignment/BeaverTails", split=split)
        dataset = dataset.map(translate)

        if args.save_to_disk:
            DATA_PATH_DIR = Path("data", "BeaverTails-it", split)
            DATA_PATH_DIR.mkdir(exist_ok=True, parents=True)
            dataset.save_to_disk(DATA_PATH_DIR)

        if args.push_to_hub:
            dataset.push_to_hub("saiteki-kai/BeaverTails-it", split=split)
