"""Minimal LLM client: Anthropic API or any OpenAI-compatible endpoint (Ollama, Groq, OpenRouter...).

Environment variables:
  LLM_PROVIDER   "anthropic" (default) or "openai"
  LLM_MODEL      model name, e.g. "claude-opus-5-5", "qwen3:8b"
  ANTHROPIC_API_KEY / OPENAI_API_KEY, and OPENAI_BASE_URL for compatible servers
"""
import json
import os
import re

import requests


def complete(prompt, max_tokens=1500):
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    model = os.getenv("LLM_MODEL", "claude-opus-5-5")
    if provider == "anthropic":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'none')}"},
        json={"model": model, "temperature": 0, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json(text):
    """Extract the first JSON object from a model answer."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in answer: {text[:200]}")
    return json.loads(m.group(0))
