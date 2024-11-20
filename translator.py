import typing as t

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


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

    @t.override
    def _prepare_prompt(self, prompt: str) -> str:
        chat_template = [{"role": "user", "content": prompt}]

        return self.tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)  # type: ignore  # noqa: PGH003

    def post_process(self, text: str) -> str:
        if "[/INST]" in text:
            return text.split("[/INST]")[1]

        return text
