"""Shared NVIDIA API client (OpenAI-compatible) for SDG + judging.

Key is read from a gitignored .env (NVIDIA_API_KEY). Get a free key at
https://build.nvidia.com. Model id is config.TEACHER_MODEL.
"""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from config import NVIDIA_BASE_URL

load_dotenv()  # read .env into os.environ if present

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def get_client() -> OpenAI:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("Set NVIDIA_API_KEY in .env (see .env.example).")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)


def chat(client: OpenAI, prompt: str, *, system: str = "",
         temperature: float = 0.7, max_tokens: int = 512) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=os.environ.get("TEACHER_MODEL_OVERRIDE") or _default_model(),
        messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _default_model() -> str:
    from config import TEACHER_MODEL
    return TEACHER_MODEL


def extract_json(text: str) -> dict | None:
    """First balanced {...} block that parses as JSON, else None."""
    text = _THINK.sub("", text or "")
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
    return None
