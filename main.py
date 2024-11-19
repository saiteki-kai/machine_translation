import typing as t

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


MODEL_NAME = "haoranxu/X-ALMA-13B-Group2"

TRANSLATION_PROMPT = "Translate this from English to Italian:\nEnglish: {text}\nItalian:"


def prepare_prompt(text: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> str:
    chat_template = [
        {"role": "user", "content": TRANSLATION_PROMPT.format(text=text)},
    ]

    return tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)  # type: ignore  # noqa: PGH003


def prepare_data(
    text: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt = prepare_prompt(text, tokenizer)
    encoded_inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=max_length, truncation=True)

    return encoded_inputs.input_ids.cuda()


def translate(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int = 512,
    generation_config: GenerationConfig | None = None,
    post_processing: t.Callable[[str], str] | None = None,
) -> str:
    input_ids = prepare_data(text, tokenizer, max_length=max_length)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, generation_config=generation_config)
        decoded = tokenizer.decode(outputs, skip_special_tokens=True)

    if post_processing is None:
        return decoded

    return post_processing(decoded)


def post_process(text: str) -> str:
    if "[/INST]" in text:
        return text.split("[/INST]")[1]

    return text


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model = torch.compile(model, mode="reduce-overhead")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    gen_config = GenerationConfig(num_beams=5, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9)

    dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_test[:1%]")

    translated_dataset = dataset.map(
        lambda x: translate(x["prompt"], model, tokenizer, generation_config=gen_config),  # type: ignore  # noqa: PGH003
    )
