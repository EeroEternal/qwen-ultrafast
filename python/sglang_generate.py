#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    body = {
        "input_ids": [9419],
        "sampling_params": {
            "max_new_tokens": args.tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
        },
    }
    for run in range(args.runs):
        request = urllib.request.Request(
            f"{args.url}/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["run"] = run
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
