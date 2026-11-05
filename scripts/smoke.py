"""Smoke test for a running API: health, then one English and one German question.

    python scripts/smoke.py http://localhost:8080 [--vector-store qdrant]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

QUESTIONS = [
    ("What is the Wi-Fi password?", "en"),
    ("Kann ich die Mitgliedschaft pausieren, wenn ich krank bin?", "de"),
]


def _call(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--vector-store", help="fail unless /health reports this vector store")
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")

    health = _call(f"{base}/health")
    print(f"health: {health}")
    if args.vector_store and health["vector_store"] != args.vector_store:
        print(f"FAIL: vector_store is {health['vector_store']!r}, expected {args.vector_store!r}")
        return 1
    for question, lang in QUESTIONS:
        r = _call(f"{base}/chat", {"question": question})
        print(f"{question} -> [{r['mode']}, {r['lang']}] {r['answer']}")
        if r["lang"] != lang or not r["citations"]:
            print(f"FAIL: expected an answer in {lang!r} with at least one citation")
            return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
