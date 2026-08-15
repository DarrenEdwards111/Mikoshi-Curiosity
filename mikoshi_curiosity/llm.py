"""Typed LLM generation for open-ended research conjectures."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Callable, Mapping, Optional, Protocol, Sequence, Tuple

from mikoshi_curiosity.research import Concept, Conjecture


class TextProvider(Protocol):
    def complete(self, prompt: str) -> str: ...


class CallableTextProvider:
    def __init__(self, complete: Callable[[str], str]):
        self._complete = complete

    def complete(self, prompt: str) -> str:
        return self._complete(prompt)


class JSONHTTPProvider:
    """Small dependency-free base class for JSON model APIs."""

    def __init__(self, url: str, headers: Optional[Mapping[str, str]] = None,
                 timeout: float = 120.0,
                 transport: Optional[Callable[[str, Mapping[str, str], Mapping[str, object], float], Mapping[str, object]]] = None):
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.transport = transport or self._request

    @staticmethod
    def _request(url: str, headers: Mapping[str, str], payload: Mapping[str, object], timeout: float) -> Mapping[str, object]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model API returned HTTP {exc.code}: {detail}") from exc


class OllamaProvider(JSONHTTPProvider):
    def __init__(self, model: str = "gpt-oss:20b", base_url: str = "http://localhost:11434",
                 timeout: float = 120.0, transport=None):
        super().__init__(f"{base_url.rstrip('/')}/api/generate", timeout=timeout, transport=transport)
        self.model = model

    def complete(self, prompt: str) -> str:
        result = self.transport(self.url, self.headers, {
            "model": self.model, "prompt": prompt, "stream": False,
            "format": "json",
        }, self.timeout)
        return str(result["response"])


class OpenAIProvider(JSONHTTPProvider):
    def __init__(self, api_key: str, model: str = "gpt-5-mini", base_url: str = "https://api.openai.com/v1",
                 timeout: float = 120.0, transport=None):
        super().__init__(f"{base_url.rstrip('/')}/chat/completions", {"Authorization": f"Bearer {api_key}"}, timeout, transport)
        self.model = model

    def complete(self, prompt: str) -> str:
        result = self.transport(self.url, self.headers, {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }, self.timeout)
        return str(result["choices"][0]["message"]["content"])


class AnthropicProvider(JSONHTTPProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5", base_url: str = "https://api.anthropic.com/v1",
                 timeout: float = 120.0, transport=None):
        super().__init__(f"{base_url.rstrip('/')}/messages", {
            "x-api-key": api_key, "anthropic-version": "2023-06-01",
        }, timeout, transport)
        self.model = model

    def complete(self, prompt: str) -> str:
        result = self.transport(self.url, self.headers, {
            "model": self.model, "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }, self.timeout)
        return "".join(str(block.get("text", "")) for block in result["content"] if block.get("type") == "text")


def _json_object(text: str) -> Mapping[str, object]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model output did not contain a JSON object")
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def _strings(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("typed conjecture list fields must contain strings")
    return tuple(value)


class LLMConjectureGenerator:
    """Ask a model for new typed conjectures and validate its JSON output."""

    def __init__(self, provider: TextProvider, failure_memory=None):
        self.provider = provider
        self.failure_memory = failure_memory

    def _prompt(self, seed: Conjecture, concepts: Sequence[Concept], n: int) -> str:
        concept_text = "\n".join(f"- {item.name}: {item.description}" for item in concepts)
        failures = () if self.failure_memory is None else self.failure_memory.similar_failures(seed.text, limit=5)
        failure_text = "\n".join(f"- {item['name']}: {item['reason']}" for item in failures) or "- none"
        return f"""You are generating research conjectures, not claiming proofs.
Return one JSON object with key \"conjectures\" containing at most {n} objects.
Each object must have exactly these typed fields:
name: string; statement: string; definitions: string[]; assumptions: string[];
proof_sketch: string[]; tags: string[].

Seed conjecture:
{seed.text}

Nearby concepts:
{concept_text or '- none'}

Previously rejected nearby routes (do not repeat their hidden assumption):
{failure_text}

Invent genuinely different definitions, invariants, proof decompositions, or
counterexample routes. State every load-bearing assumption explicitly. JSON only."""

    def generate(self, seed: Conjecture, concepts: Sequence[Concept], n: int) -> Sequence[Conjecture]:
        payload = _json_object(self.provider.complete(self._prompt(seed, concepts, n)))
        items = payload.get("conjectures", [])
        if not isinstance(items, list):
            raise ValueError("conjectures must be a JSON list")
        generated = []
        for raw in items[:max(0, n)]:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not isinstance(raw.get("statement"), str):
                raise ValueError("each conjecture requires string name and statement")
            generated.append(Conjecture(
                name=raw["name"], statement=raw["statement"],
                definitions=_strings(raw.get("definitions")),
                assumptions=_strings(raw.get("assumptions")),
                proof_sketch=_strings(raw.get("proof_sketch")),
                tags=_strings(raw.get("tags")),
                provenance=seed.provenance + ("llm-generated",),
                parent_id=seed.id, generation=seed.generation + 1,
            ))
        return generated


class LLMLeanRepairer:
    """Turn Lean diagnostics into a revised source candidate."""

    def __init__(self, provider: TextProvider):
        self.provider = provider

    def __call__(self, conjecture: Conjecture, source: str, error: str, attempt: int) -> str:
        response = self.provider.complete(f"""Repair this Lean 4 source. Preserve the theorem statement and do not use sorry,
admit, custom axioms, or unsafe. Return Lean source only.
Conjecture: {conjecture.statement}
Attempt: {attempt}
Compiler diagnostics:\n{error}\nSource:\n{source}""")
        return re.sub(r"^```(?:lean)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
