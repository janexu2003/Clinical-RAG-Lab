"""
One tiny interface for every LLM:  llm.complete(system, user, json_mode=False) -> str

    get_llm("openai:gpt-4o-mini")     # needs OPENAI_API_KEY
    get_llm("ollama:llama3.1:8b")     # local, free, private (https://ollama.com)
    get_llm("none")                   # retrieval-only experiments

Calls and tokens are counted so the evaluation can report cost next to quality.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, Optional


class LLM:
    name = "base"

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str) -> Dict:
        return parse_json(self.complete(system, user, json_mode=True))

    def usage(self) -> Dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


class OpenAIChat(LLM):
    def __init__(self, model: str):
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def complete(self, system, user, json_mode=False, temperature=0.0):
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
        self.calls += 1
        if r.usage:
            self.prompt_tokens += r.usage.prompt_tokens
            self.completion_tokens += r.usage.completion_tokens
        return r.choices[0].message.content or ""


class OllamaChat(LLM):
    def __init__(self, model: str, host: Optional[str] = None):
        super().__init__()
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama:{model}"

    def complete(self, system, user, json_mode=False, temperature=0.0):
        import requests
        body = {"model": self.model, "stream": False, "options": {"temperature": temperature},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=300)
        r.raise_for_status()
        data = r.json()
        self.calls += 1
        self.prompt_tokens += data.get("prompt_eval_count", 0)
        self.completion_tokens += data.get("eval_count", 0)
        return data["message"]["content"]


class StubLLM(LLM):
    """For tests: a python function decides the reply."""

    def __init__(self, fn: Callable[[str, str], str]):
        super().__init__()
        self.fn = fn
        self.name = "stub"

    def complete(self, system, user, json_mode=False, temperature=0.0):
        self.calls += 1
        return self.fn(system, user)


def get_llm(spec: Optional[str]) -> Optional[LLM]:
    if not spec or spec == "none":
        return None
    if spec == "dummy":
        # smoke tests only: always answers option A / empty JSON, costs nothing
        return StubLLM(lambda system, user: '{"answer_choice": "A", "queries": [], "verdict": "sufficient"}')
    provider, _, model = spec.partition(":")
    if provider == "openai":
        return OpenAIChat(model or "gpt-4o-mini")
    if provider == "ollama":
        return OllamaChat(model or "llama3.1:8b")
    raise ValueError(f"unknown llm spec {spec!r} (use openai:<model> or ollama:<model>)")


def parse_json(text: str) -> Dict:
    """Robust JSON extraction (models sometimes wrap JSON in prose or ``` fences)."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return {}
