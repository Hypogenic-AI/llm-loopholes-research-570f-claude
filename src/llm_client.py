"""Thin OpenRouter client with retries, caching, and reasoning-model handling.

OpenRouter exposes both regular and "reasoning" model outputs through the same
chat-completions endpoint. For long-CoT models (DeepSeek R1 distills), the
externally-visible answer can be located in `message.reasoning` when the model
exhausts its budget before producing a `content` token.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

CACHE_DIR = Path("results/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_KEY"],
            timeout=120.0,
        )
    return _client


@dataclass
class LLMResponse:
    model: str
    content: str  # extracted answer text (handles reasoning models)
    raw_content: Optional[str]
    reasoning: Optional[str]
    usage: dict
    finish_reason: Optional[str]
    seed: Optional[int]


def _cache_key(model: str, messages, temperature, top_p, max_tokens, seed) -> str:
    h = hashlib.sha256()
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        },
        sort_keys=True,
    ).encode()
    h.update(payload)
    return h.hexdigest()


def _coerce_response(model: str, resp, seed) -> LLMResponse:
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning", None)
    # If model is a reasoning model that exhausted tokens during CoT, take reasoning text
    if (not content) and reasoning:
        content = reasoning
    usage = {}
    try:
        usage = resp.usage.model_dump() if resp.usage else {}
    except Exception:
        usage = {}
    return LLMResponse(
        model=model,
        content=content or "",
        raw_content=msg.content,
        reasoning=reasoning,
        usage=usage,
        finish_reason=getattr(resp.choices[0], "finish_reason", None),
        seed=seed,
    )


class TransientError(Exception):
    """Wrap retryable exceptions for tenacity."""


@retry(
    retry=retry_if_exception_type(TransientError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call(model, messages, temperature, top_p, max_tokens, seed) -> LLMResponse:
    try:
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = get_client().chat.completions.create(**kwargs)
        return _coerce_response(model, resp, seed)
    except Exception as e:
        # Treat all errors as transient for retry (except clear 4xx)
        emsg = str(e).lower()
        non_retryable = ("authentication", "invalid_api_key", "unauthorized")
        if any(s in emsg for s in non_retryable):
            raise
        raise TransientError(str(e)) from e


def query(
    model: str,
    user: str,
    *,
    system: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 600,
    seed: int | None = None,
    use_cache: bool = True,
) -> LLMResponse:
    """Single chat-completion call with disk cache."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    key = _cache_key(model, messages, temperature, top_p, max_tokens, seed)
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        try:
            d = json.loads(cache_file.read_text())
            return LLMResponse(**d)
        except Exception:
            cache_file.unlink(missing_ok=True)

    t0 = time.time()
    resp = _call(model, messages, temperature, top_p, max_tokens, seed)
    elapsed = time.time() - t0
    # Persist
    out = {
        "model": resp.model,
        "content": resp.content,
        "raw_content": resp.raw_content,
        "reasoning": resp.reasoning,
        "usage": {**resp.usage, "elapsed_s": elapsed},
        "finish_reason": resp.finish_reason,
        "seed": resp.seed,
    }
    cache_file.write_text(json.dumps(out))
    return LLMResponse(**out)


# Constants used by experiments
SUBJECT_MODELS = {
    # short id -> openrouter id
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "deepseek-r1-70b": "deepseek/deepseek-r1-distill-llama-70b",
}

MODEL_TIER = {
    "gpt-4o-mini": "frontier",
    "claude-3.5-haiku": "frontier",
    "llama-3.1-70b": "large_open",
    "llama-3.1-8b": "small_open",
    "deepseek-r1-70b": "reasoning",
}

JUDGE_MODELS = {
    "judge-gpt-4o-mini": "openai/gpt-4o-mini",
    "judge-claude-haiku": "anthropic/claude-3.5-haiku",
}
