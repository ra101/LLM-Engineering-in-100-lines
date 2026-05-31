"""Interactive LLM chat CLI with streaming and token stats."""

import argparse
from colorama import Style
from utils import chat_with_stats, stream_chat_with_stats


SYSTEM_PROMPT = (
    'Explain things as if talking to a 5-year-old. '
    'Keep it to 2 sentences at max. DO NOT SAY BAD WORDS.'
)


def print_stats(stats):
    """Display token usage after each response."""
    print(
        f"{Style.DIM}  "
        f"[{stats['input_tokens']} in → {stats['output_tokens']} out"
        f" | {stats['tokens_per_sec']:.1f} tok/s] {Style.RESET_ALL}"
    )


def respond(messages, stream=True):
    """Get LLM response (streaming or complete), print stats, return text."""
    if not stream:
        content, stats = chat_with_stats(messages)
        print(f"Assistant: {Style.BRIGHT} {content}  {Style.RESET_ALL}")
        print_stats(stats)
        return content

    print(f"Assistant: {Style.BRIGHT} ", end="", flush=True)
    tokens = []
    stats = None
    for token in stream_chat_with_stats(messages):
        if isinstance(token, dict):
            stats = token
        else:
            tokens.append(token)
            print(token, end="", flush=True)
    print(Style.RESET_ALL)
    if stats:
        print_stats(stats)
    return "".join(tokens)


def mode_single(stream=True):
    """One question, one answer, done."""
    print("Mode: single question → single answer (no system prompt, no history)")
    user_input = input("You: ").strip()
    if not user_input:
        return
    messages = [{"role": "user", "content": user_input}]
    respond(messages, stream=stream)


def mode_system(stream=True):
    """one question, one answer with system prompt, done."""
    print("Mode: single question → single answer, with system prompt (no history)")
    system = input("System prompt (press enter for default): ").strip()
    system = system or SYSTEM_PROMPT
    print(f"System: {system}\n")

    user_input = input("You: ").strip()
    if not user_input:
        return
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
    respond(messages, stream=stream)


def mode_chat(stream=True):
    """Multi-turn chat with system prompt and full history."""
    print("Mode: multi-turn chat with system prompt and full history")
    print("Type 'exit' or 'quit' to end the chat.\n")
    system = input("System prompt (press enter for default): ").strip()
    system = system or SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]
    print(f"System: {system}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": user_input})
        content = respond(messages, stream=stream)
        messages.append({"role": "assistant", "content": content})
        print()


def get_parser():
    parser = argparse.ArgumentParser(description="Chat with a local LLM")
    parser.add_argument(
        "--mode", choices=["single", "system", "chat"], default="single",
        help=(
            'single: one question, one answer, done.\n'
            'system: one question, one answer with system prompt, done.\n'
            'chat: multi-turn chat with system prompt and full history, '
            'until user exits.'
        ),
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="disable streaming.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    mode, stream = args.mode, not args.no_stream

    if mode == "system":
        return mode_system(stream=stream)
    elif mode == "chat":
        return mode_chat(stream=stream)

    return mode_single(stream=stream)


if __name__ == "__main__":
    main()
