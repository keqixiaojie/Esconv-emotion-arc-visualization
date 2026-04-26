#!/usr/bin/env python3
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app_esconv import _compute_sync_dataset, _sync_cache_path  # noqa: E402


def parse_int_list(value: str):
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def parse_str_list(value: str):
    return [part.strip() for part in value.split(',') if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Precompute synchrony-range caches for ESConv visualization.")
    parser.add_argument(
        "--tail-pcts",
        default="0,5,15,25,35",
        help="Comma-separated tail percentages to precompute. Default: 0,5,15,25,35")
    parser.add_argument(
        "--window-sizes",
        default="2",
        help="Comma-separated window sizes. Default: 2")
    parser.add_argument(
        "--smooth-modes",
        default="avg",
        help="Comma-separated smooth modes. Default: avg")
    parser.add_argument(
        "--granularities",
        default="word",
        help="Comma-separated granularities. Default: word")
    args = parser.parse_args()

    tail_pcts = parse_int_list(args.tail_pcts)
    window_sizes = parse_int_list(args.window_sizes)
    smooth_modes = parse_str_list(args.smooth_modes)
    granularities = parse_str_list(args.granularities)

    jobs = []
    for tail_pct in tail_pcts:
        for ws in window_sizes:
            for smooth_mode in smooth_modes:
                for granularity in granularities:
                    actual_mode = 'context' if smooth_mode == 'context' and granularity == 'sentence' else 'avg'
                    jobs.append((tail_pct, ws, actual_mode, granularity))

    print(f"Preparing {len(jobs)} synchrony cache job(s)...")
    for idx, (tail_pct, ws, smooth_mode, granularity) in enumerate(jobs, start=1):
        tail_ratio = tail_pct / 100.0
        cache_path = _sync_cache_path(tail_ratio, ws, smooth_mode, granularity)
        started = time.time()
        dataset = _compute_sync_dataset(tail_ratio, ws, smooth_mode, granularity, compute_if_missing=True)
        elapsed = time.time() - started
        if dataset is None:
            print(f"[{idx}/{len(jobs)}] FAILED  tail={tail_pct}% ws={ws} mode={smooth_mode} gran={granularity}")
            continue
        print(
            f"[{idx}/{len(jobs)}] OK  tail={tail_pct}% ws={ws} mode={smooth_mode} gran={granularity} "
            f"points={dataset['point_count']} convs={dataset['used_conversations']} "
            f"time={elapsed:.1f}s cache={cache_path}"
        )


if __name__ == "__main__":
    main()
