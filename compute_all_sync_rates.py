#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app_esconv import (  # noqa: E402
    CACHE_DIR,
    SYNC_DEFAULT_GRANULARITY,
    SYNC_DEFAULT_SMOOTH_MODE,
    SYNC_DEFAULT_WINDOW_SIZE,
    conv_ids,
    _default_diff_cache_path,
    _load_default_diff_bundle,
    _mahalanobis_inside,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_default_sync_series(conv_id: int):
    bundle = _load_default_diff_bundle(conv_id, compute_if_missing=True)
    if not bundle or not bundle.get("current_sync"):
        return None
    current = bundle["current_sync"]
    return {
        "conv_id": conv_id,
        "points": np.asarray(current["points"], dtype=float),
        "turns": list(current["turns"]),
        "utterance_spans": np.asarray(current["utterance_spans"], dtype=float),
        "cache_path": _default_diff_cache_path(conv_id),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute synchrony rates for all conversations from cached default diff arcs.")
    parser.add_argument("--tail-pct", type=int, default=25, help="Tail percentage. Default: 25")
    parser.add_argument("--confidence-pct", type=int, default=67, help="Ellipsoid confidence percentage. Default: 67")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(CACHE_DIR, "sync_rates"),
        help="Directory for CSV/JSON outputs. Default: src/cache/sync_rates",
    )
    args = parser.parse_args()

    tail_ratio = float(args.tail_pct) / 100.0
    confidence_ratio = float(args.confidence_pct) / 100.0
    ensure_dir(args.output_dir)

    started = time.time()
    series_all = []
    for idx, conv_id in enumerate(conv_ids, start=1):
        item = load_default_sync_series(conv_id)
        if item is None or len(item["points"]) == 0:
            continue
        series_all.append(item)
        if idx % 100 == 0:
            print(f"loaded {idx}/{len(conv_ids)} bundles | kept {len(series_all)}", flush=True)

    dataset_points = []
    per_conv_current = {}
    for item in series_all:
        size = len(item["points"])
        if size <= 0:
            continue
        if tail_ratio <= 0:
            start_idx = 0
        else:
            tail_count = max(1, int(np.ceil(size * tail_ratio)))
            start_idx = max(0, size - tail_count)
        tail_points = np.asarray(item["points"][start_idx:], dtype=float)
        if len(tail_points) == 0:
            continue
        per_conv_current[item["conv_id"]] = {
            "points": np.asarray(item["points"], dtype=float),
            "spans": np.asarray(item["utterance_spans"], dtype=float),
            "turns": list(item["turns"]),
        }
        dataset_points.append(tail_points)

    if not dataset_points:
        raise RuntimeError("No tail points available. Default diff bundles may be missing or empty.")

    all_points = np.vstack(dataset_points)
    mean = np.mean(all_points, axis=0)
    cov = np.cov(all_points, rowvar=False) if len(all_points) > 1 else np.eye(3) * 1e-6
    cov = np.asarray(cov, dtype=float) + np.eye(3) * 1e-6
    chi2_threshold = float(chi2.ppf(confidence_ratio, df=3))

    rows = []
    for conv_id in sorted(per_conv_current.keys()):
        item = per_conv_current[conv_id]
        inside, dist2 = _mahalanobis_inside(item["points"], mean, cov, chi2_threshold)
        total_span = float(np.sum(item["spans"]))
        inside_span = float(np.sum(item["spans"] * inside.astype(float)))
        sync_rate = (inside_span / total_span) if total_span > 0 else 0.0
        rows.append({
            "conv_id": conv_id,
            "sync_rate": sync_rate,
            "inside_span": inside_span,
            "total_span": total_span,
            "total_points": int(len(item["points"])),
            "inside_points": int(np.sum(inside)),
            "mean_dist2": float(np.mean(dist2)) if len(dist2) else 0.0,
        })

    stem = f"sync_rates_tail{args.tail_pct}_conf{args.confidence_pct}_default_sentence_context_ws2"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    json_path = os.path.join(args.output_dir, f"{stem}.json")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["conv_id", "sync_rate", "inside_span", "total_span", "total_points", "inside_points", "mean_dist2"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "tail_pct": args.tail_pct,
        "confidence_pct": args.confidence_pct,
        "default_mode": {
            "granularity": SYNC_DEFAULT_GRANULARITY,
            "smooth_mode": SYNC_DEFAULT_SMOOTH_MODE,
            "window_size": SYNC_DEFAULT_WINDOW_SIZE,
        },
        "global_distribution": {
            "mean": mean.tolist(),
            "cov": cov.tolist(),
            "chi2_threshold": chi2_threshold,
            "conversation_count": len(rows),
            "point_count": int(len(all_points)),
        },
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} rows.", flush=True)
    print(f"CSV:  {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
