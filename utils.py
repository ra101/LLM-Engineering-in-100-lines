"""Shared Chat utilities"""

import os

from ollama import chat as ollama_chat

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")


def chat_with_stats(messages, temperature=0.7):
    """Send messages to Ollama and return (content, stats_dict)."""
    data = ollama_chat(
        model=MODEL, messages=messages,
        options={"temperature": temperature},
    )
    stats = {
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "tokens_per_sec": (data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9)) if data.get("eval_duration") else 0,
    }
    return data["message"]["content"], stats


def stream_chat_with_stats(messages, temperature=0.7):
    """Stream tokens from Ollama. Last yielded item is a stats dict."""
    for chunk in ollama_chat(
        model=MODEL, messages=messages,
        stream=True, options={"temperature": temperature},
    ):
        if chunk.get("done"):
            yield {
                "input_tokens": chunk.get("prompt_eval_count", 0),
                "output_tokens": chunk.get("eval_count", 0),
                "tokens_per_sec": (chunk.get("eval_count", 0) / (chunk.get("eval_duration", 1) / 1e9)) if chunk.get("eval_duration") else 0,
            }
            return
        yield chunk["message"]["content"]
