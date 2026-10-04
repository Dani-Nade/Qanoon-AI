"""Minimal client for a local Ollama server (chat completion, streaming or not)."""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging

import requests

LOGGER = logging.getLogger(__name__)


class GenerationLimitReached(RuntimeError):
    """The token limit was used up (typically by thinking) before any answer text was produced."""


class OllamaClient:
    def __init__(self, base_url: str, model: str, *, num_ctx: int = 16384, think: bool = False,
                 max_tokens: int | None = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.max_tokens = max_tokens
        self.timeout = timeout

    def status(self) -> dict:
        """{"reachable": bool, "model_available": bool}; never raises."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            names = {model.get("name") for model in response.json().get("models", [])}
            return {"reachable": True, "model_available": self.model in names}
        except (requests.RequestException, ValueError):
            return {"reachable": False, "model_available": False}

    def _payload(self, messages: list[dict], *, stream: bool, temperature: float, max_tokens: int | None, think: bool) -> dict:
        options = {"temperature": temperature, "num_ctx": self.num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens  # counts thinking tokens too
        return {"model": self.model, "messages": messages, "stream": stream, "think": think, "options": options}

    def complete(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int | None = None,
                 think: bool = False, json_output: bool = False) -> str:
        payload = self._payload(messages, stream=False, temperature=temperature, max_tokens=max_tokens, think=think)
        if json_output:
            payload["format"] = "json"
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream(self, messages: list[dict], *, temperature: float = 0.3, think: bool | None = None) -> Iterator[tuple[str, str]]:
        """Yield ("thinking", text) chunks of the model's reasoning, then ("content", text) answer chunks.

        Raises GenerationLimitReached if the token limit ends generation before any answer text.
        """
        think = self.think if think is None else think
        with requests.post(
            f"{self.base_url}/api/chat",
            json=self._payload(messages, stream=True, temperature=temperature, max_tokens=self.max_tokens, think=think),
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            produced = False
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                message = event.get("message", {})
                if message.get("thinking"):
                    yield "thinking", message["thinking"]
                if message.get("content"):
                    produced = True
                    yield "content", message["content"]
                if event.get("done"):
                    if not produced and event.get("done_reason") == "length":
                        raise GenerationLimitReached("token limit reached before an answer was written")
                    return
