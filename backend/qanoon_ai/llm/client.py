"""LLM client interface."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from threading import Lock

from qanoon_ai.llm.attention import is_fatal_cuda_error, is_recoverable_oom, register_attention
from qanoon_ai.llm.prompts import ABSTENTION, evidence_messages

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str


class DisabledLLMClient:
    model_name = "disabled"
    ready = False

    def generate(self, prompt: str) -> LLMResult:
        return LLMResult(text="", model=self.model_name)


def adapter_complete(path: Path) -> bool:
    required = ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "training-report.json")
    if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
        return False
    try:
        report = json.loads((path / "training-report.json").read_text(encoding="utf-8"))
        return bool(report.get("completed_at") and report.get("base_model") and report.get("base_revision"))
    except (OSError, ValueError):
        return False


def validate_quotation(text: str, excerpts: list[str]) -> bool:
    """Validate the entire response, not just the presence of a citation number."""
    if text.strip() == ABSTENTION:
        return True
    match = re.fullmatch(r'["“](.+)["”]\s*\[(\d+)\]', text.strip(), re.DOTALL)
    if not match:
        return False
    quote, number = match.groups()
    index = int(number) - 1
    normalized = " ".join(quote.split())
    return (
        0 <= index < len(excerpts) and len(normalized) >= 20
        and normalized in " ".join(excerpts[index].split())
    )


class LocalLLMClient:
    """Load the completed local adapter; no model download during API requests."""

    model_name = "qanoon-local-lora"

    def __init__(self, model_dir: Path, cache_dir: Path):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self._model = self._tokenizer = None
        self._lock = Lock()
        self._failed = False

    @property
    def ready(self) -> bool:
        if self._failed or not adapter_complete(self.model_dir):
            return False
        try:
            self.load()
        except Exception:
            LOGGER.exception("Local Qanoon model could not be loaded")
            self._failed = True
            return False
        return True

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            if not adapter_complete(self.model_dir):
                raise RuntimeError("A completed local adapter is required")
            import torch
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            report = json.loads((self.model_dir / "training-report.json").read_text(encoding="utf-8"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            # Resolve the pinned snapshot first. Transformers' adapter discovery
            # can otherwise ignore local_files_only/cache_dir for a Hub model ID.
            base_path = snapshot_download(
                report["base_model"], revision=report["base_revision"],
                cache_dir=str(self.cache_dir), local_files_only=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_path, local_files_only=True,
                dtype=dtype, attn_implementation=register_attention(),
            )
            model = PeftModel.from_pretrained(base, self.model_dir, local_files_only=True)
            model.to(device).eval()
            model.config.use_cache = True
            self._tokenizer, self._model = tokenizer, model

    def answer(self, question: str, sources: list[tuple[str, str]]) -> LLMResult:
        if not sources:
            return LLMResult(ABSTENTION, self.model_name)
        if not self.ready:
            raise RuntimeError("Local model is not ready")
        import torch

        with self._lock, torch.inference_mode():
            # Keep the same numbering as the citations shown in the app.
            sources = sources[:3]
            prompt = self._tokenizer.apply_chat_template(
                evidence_messages(question, sources), tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            if inputs.input_ids.shape[-1] > 3072:
                raise ValueError("Question and sources exceed the local model input budget")
            try:
                outputs = self._model.generate(
                    **inputs, max_new_tokens=192, do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            except Exception as error:
                if is_recoverable_oom(error):
                    torch.cuda.empty_cache()
                elif is_fatal_cuda_error(error):
                    self._failed = True
                    LOGGER.critical("CUDA context is unusable; restart the backend to re-enable the local model")
                raise
            text = self._tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            if not validate_quotation(text, [excerpt for _, excerpt in sources]):
                raise ValueError("Generated quotation did not match its cited source")
            return LLMResult(text, self.model_name)
