import typing as t

import torch

from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from utils import get_language_name, get_xalma_model_name_by_group, get_xalma_model_name_by_lang


class Translator:
    model: PreTrainedModel | t.Callable
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        self.model = torch.compile(self.model, mode="reduce-overhead")

    def _prepare_prompt(self, prompt: str) -> str:
        return prompt

    def _prepare_data(self, text: str, max_length: int = 512) -> dict[str, torch.Tensor]:
        prompt = self._prepare_prompt(text)

        encoded_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        )

        return encoded_inputs.input_ids.cuda()

    def translate(
        self,
        text: str,
        max_length: int = 512,
        generation_config: GenerationConfig | None = None,
        translation_prompt: t.Callable[[str], str] | None = None,
    ) -> str:
        if translation_prompt is None:
            input_ids = self._prepare_data(text, max_length=max_length)
        else:
            prompt = translation_prompt(text)
            input_ids = self._prepare_data(prompt, max_length=max_length)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, generation_config=generation_config)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return self.post_processing(decoded)

    def post_processing(self, text: str) -> str:
        return text


class ALMATranslator(Translator):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    @staticmethod
    def from_target_lang(target_lang: str):
        model_name = get_xalma_model_name_by_lang(target_lang)
        return ALMATranslator(model_name)

    @staticmethod
    def from_group(group_id: int):
        model_name = get_xalma_model_name_by_group(group_id)
        return ALMATranslator(model_name)

    @t.override
    def _prepare_prompt(self, prompt: str) -> str:
        chat_template = [{"role": "user", "content": prompt}]

        return self.tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)  # type: ignore  # noqa: PGH003

    def post_process(self, text: str) -> str:
        if "[/INST]" in text:
            return text.split("[/INST]")[1]

        return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate using the ALMA model.")
    parser.add_argument("--model-name", type=str, default="haoranxu/ALMA-7B", help="Huggingface model.")
    parser.add_argument("--source-lang", type=str, default="en", help="Source language.", required=True)
    parser.add_argument("--target-lang", type=str, default="it", help="Target language.", required=True)
    parser.add_argument("--dataset-name", type=str, default="PKU-Alignment/BeaverTails", help="Huggingface dataset.")
    parser.add_argument("--dataset-split", type=str, default="330k_test", help="Dataset split to use.")
    parser.add_argument("--num-beams", type=int, default=5, help="Number of beams for beam search.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum length for tokenized input.")

    args = parser.parse_args()

    model = ALMATranslator.from_target_lang(args.target_lang)

    gen_config = GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    def get_translation_prompt(source_lang: str, target_lang: str) -> t.Callable[[str], str]:
        src_name = get_language_name(source_lang)
        tgt_name = get_language_name(target_lang)

        return lambda text: f"Translate this from {src_name} to {tgt_name}:\n{src_name}: {text}\n{tgt_name}:"

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    for example in tqdm(dataset):
        prompt = model.translate(
            example["prompt"],
            max_length=args.max_length,
            generation_config=gen_config,
            translation_prompt=get_translation_prompt(args.source_lang, args.target_lang),
        )
        response = model.translate(
            example["response"],
            max_length=args.max_length,
            generation_config=gen_config,
            translation_prompt=get_translation_prompt(args.source_lang, args.target_lang),
        )

        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
