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

    def __init__(self, model_name: str, post_processing: Callable[[str], str] | None = None) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self._post_processing = post_processing

    def _prepare_data(self, text: str, max_length: int = 512) -> dict[str, torch.Tensor]:
        encoded_inputs = self._tokenizer(
            text,
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

        if self._post_processing is None:
            return decoded

        return self._post_processing(decoded)

    def compile(self, mode: str = "reduce-overhead"):
        self._model = torch.compile(self._model, mode=mode)


class ConversationalTranslator(Translator):
    def __init__(self, model_name: str, post_processing: Callable[[str], str] | None = None) -> None:
        super().__init__(model_name, post_processing)

    def _prepare_prompt(self, conversation: list[dict[str, str]] | str) -> str:
        if isinstance(conversation, str):
            conversation = [{"role": "user", "content": conversation}]

        chat_prompt = self._tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        return typing.cast(str, chat_prompt)

    def translate(
        self,
        conversation: list[dict[str, str]] | str,
        max_length: int = 512,
        generation_config: GenerationConfig | None = None,
        translation_prompt: Callable[[str], str] | None = None,
    ) -> str:
        text = self._prepare_prompt(conversation)

        return super().translate(text, max_length, generation_config, translation_prompt)
