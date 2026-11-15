#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def tokens(path, key):
    result = json.loads(Path(path).read_text())
    values = result[key]
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"{path}: {key} must be an integer token list")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-key", default="output_ids")
    parser.add_argument("--require-prefix", type=int, default=0)
    args = parser.parse_args()
    rust = tokens(args.rust, "tokens")
    reference = tokens(args.reference, args.reference_key)
    prefix = 0
    for left, right in zip(rust, reference):
        if left != right:
            break
        prefix += 1
    report = {
        "rust_tokens": len(rust),
        "reference_tokens": len(reference),
        "matching_prefix": prefix,
        "exact": rust == reference,
        "first_mismatch": None,
        "required_prefix": args.require_prefix,
    }
    if prefix < min(len(rust), len(reference)):
        report["first_mismatch"] = {
            "index": prefix,
            "rust": rust[prefix],
            "reference": reference[prefix],
        }
    report["passed"] = prefix >= args.require_prefix
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
