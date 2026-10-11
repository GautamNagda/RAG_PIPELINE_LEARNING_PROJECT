"""Ask a local Ollama model a question, streaming the answer as it is generated."""

import argparse
import os
import sys

import httpx
import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_PROMPT = "Explain RAG in simple language with one example."
DEFAULT_SYSTEM = "You are a clear, concise tutor. Use simple language."

# First request loads the model into RAM, which can be slow on CPU-only machines.
TIMEOUT_SECONDS = 120


def ask(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    system: str | None = DEFAULT_SYSTEM,
    temperature: float = 0.3,
    timeout: float = TIMEOUT_SECONDS,
):
    """Yield the reply in text pieces as the model produces them."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = ollama.Client(host=host, timeout=timeout)
    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": temperature},
    )
    for part in stream:
        piece = part.message.content
        if piece:
            yield piece


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt ('' to disable)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Asking {args.model} at {args.host} ...\n", file=sys.stderr)

    try:
        for piece in ask(
            args.prompt,
            model=args.model,
            host=args.host,
            system=args.system or None,
            temperature=args.temperature,
            timeout=args.timeout,
        ):
            print(piece, end="", flush=True)
        print()
    except (ConnectionError, httpx.ConnectError):
        print(
            f"\nCould not reach Ollama at {args.host}.\n"
            "Start it with `ollama serve` (or open the Ollama app) and try again.",
            file=sys.stderr,
        )
        return 1
    except httpx.TimeoutException:
        print(
            f"\nNo response within {args.timeout:.0f}s. The model may still be loading; "
            "retry, or raise --timeout.",
            file=sys.stderr,
        )
        return 1
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            print(
                f"\nModel '{args.model}' is not installed.\n"
                f"Run: ollama pull {args.model}",
                file=sys.stderr,
            )
        else:
            print(f"\nOllama error ({exc.status_code}): {exc.error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())