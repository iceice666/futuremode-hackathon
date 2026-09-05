#!/usr/bin/env python3
"""Quick smoke test against the vLLM OpenAI-compatible server.

Usage:
    python3 test_client.py path/to/image.jpg "What is in this image?"
"""
import base64
import sys

import requests

BASE_URL = "http://127.0.0.1:8000/v1"
SERVER_URL = f"{BASE_URL}/chat/completions"


def served_model() -> str:
    """Ask the server which model it's actually serving, so this doesn't break
    when switching between the bf16 and AWQ builds."""
    r = requests.get(f"{BASE_URL}/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <image_path> <question>", file=sys.stderr)
        sys.exit(1)

    image_path, question = sys.argv[1], sys.argv[2]
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        SERVER_URL,
        json={
            "model": served_model(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 256,
        },
        timeout=120,
    )
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
