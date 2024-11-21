import typing

from collections.abc import Callable

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
    _model: PreTrainedModel | Callable
    _tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    def __init__(self, model_name: str) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        self._model = torch.compile(self._model, mode="reduce-overhead")

    def _prepare_prompt(self, prompt: str) -> str:
        return prompt

    def _prepare_data(self, text: str, max_length: int = 512) -> dict[str, torch.Tensor]:
        prompt = self._prepare_prompt(text)

        encoded_inputs = self._tokenizer(
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
        translation_prompt: Callable[[str], str] | None = None,
    ) -> str:
        if translation_prompt is None:
            input_ids = self._prepare_data(text, max_length=max_length)
        else:
            prompt = translation_prompt(text)
            input_ids = self._prepare_data(prompt, max_length=max_length)

        with torch.no_grad():
            outputs = self._model.generate(input_ids=input_ids, generation_config=generation_config)
            decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return self.post_processing(decoded)

    def post_processing(self, text: str) -> str:
        return text


class ALMATranslator(Translator):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)

    def _prepare_prompt(self, prompt: str) -> str:
        conversation = [{"role": "user", "content": prompt}]
        chat_prompt = self._tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        return typing.cast(str, chat_prompt)

    def post_process(self, text: str) -> str:
        if "[/INST]" in text:
            return text.split("[/INST]")[1]

        return text
