#!/usr/bin/env python3
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app_esconv import (  # noqa: E402
    DEFAULT_DIFF_MEMORY_CACHE,
    conv_ids,
    _default_diff_cache_path,
    _load_default_diff_bundle,
)


def parse_int_list(value: str):
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Precompute default sentence/context/W=2 diff-arc bundles for all conversations.")
    parser.add_argument(
        "--conv-ids",
        default="",
        help="Optional comma-separated conversation ids. Default: all conversations.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete memory benefit by forcing recompute for already-cached bundles.")
    args = parser.parse_args()

    targets = parse_int_list(args.conv_ids) if args.conv_ids else list(conv_ids)
    print(f"Preparing {len(targets)} default diff bundle(s)...", flush=True)
    started = time.time()
    ok = 0
    for idx, conv_id in enumerate(targets, start=1):
        cache_path = _default_diff_cache_path(conv_id)
        if args.refresh and os.path.exists(cache_path):
            os.remove(cache_path)
            DEFAULT_DIFF_MEMORY_CACHE.pop(conv_id, None)
        item_started = time.time()
        bundle = _load_default_diff_bundle(conv_id, compute_if_missing=True)
        elapsed = time.time() - item_started
        if bundle is None:
            print(f"[{idx}/{len(targets)}] FAILED conv={conv_id}", flush=True)
            continue
        ok += 1
        print(
            f"[{idx}/{len(targets)}] OK conv={conv_id} "
            f"time={elapsed:.1f}s cache={cache_path}",
            flush=True,
        )
    print(f"Finished {ok}/{len(targets)} bundle(s) in {time.time() - started:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
