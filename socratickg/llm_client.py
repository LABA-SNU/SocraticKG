"""OpenAI-compatible LLM client and response parsing utilities."""

import json
import re
import time
from typing import Tuple, Any

from openai import OpenAI

import config


_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a cached OpenAI client, initializing it on first use."""
    global _client
    if _client is None:
        config.validate_credentials()
        _client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    return _client


def call_model(
    prompt: str,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> Tuple[str, dict]:
    """Call the configured chat model.

    Returns:
        (text, usage_dict) where usage_dict has
        {"prompt_tokens", "completion_tokens", "total_tokens"}.
    """
    client = get_client()
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            usage = dict(empty_usage)
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens
                usage["completion_tokens"] = response.usage.completion_tokens
                usage["total_tokens"] = response.usage.total_tokens

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip(), usage
            return "", usage

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[call_model] attempt {attempt + 1} failed: {e}. retrying...")
                time.sleep(retry_delay)
                continue
            print(f"[call_model] API error: {e}")
            return "", dict(empty_usage)

    return "", dict(empty_usage)


def parse_llm_json(text: str) -> list:
    """Tolerantly parse a JSON list/objects out of a possibly-messy LLM response.

    Strips markdown code fences and scans for JSON substrings, so partial
    outputs and extra prose before/after the JSON do not break parsing.
    """
    if not text:
        return []

    # Strip markdown code fences if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    text = text.strip()

    results: list[Any] = []
    decoder = json.JSONDecoder()
    while len(text) > 0:
        match = re.search(r"[\[\{]", text)
        if not match:
            break
        text = text[match.start():]
        try:
            obj, end_idx = decoder.raw_decode(text)
            if isinstance(obj, list):
                results.extend(obj)
            elif isinstance(obj, dict):
                results.append(obj)
            text = text[end_idx:].strip()
        except json.JSONDecodeError:
            text = text[1:]
    return results
