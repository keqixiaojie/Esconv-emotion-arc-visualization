#!/usr/bin/env python3
import argparse
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
    SYNC_CONFIDENCE,
    _build_sync_kde_cache,
    _compute_diff_series,
    _sample_points,
    _serialize_sync_dataset,
    _sync_cache_path,
    build_conversation_cache,
    conv_ids,
)


def parse_int_list(value: str):
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def parse_str_list(value: str):
    return [part.strip() for part in value.split(',') if part.strip()]


def _series_cache_path(ws: int, smooth_mode: str, granularity: str) -> str:
    cache_dir = os.path.dirname(_sync_cache_path(0.35, ws, smooth_mode, granularity))
    return os.path.join(cache_dir, f"fullscan_ws{ws}_{smooth_mode}_{granularity}.npz")


def save_series_cache(path: str, series_by_conv):
    lengths = np.asarray([len(series) for series in series_by_conv], dtype=np.int32)
    if len(series_by_conv) == 0:
        points = np.empty((0, 3), dtype=np.float32)
    else:
        points = np.vstack(series_by_conv).astype(np.float32, copy=False)
    np.savez_compressed(path, points=points, lengths=lengths)


def load_series_cache(path: str):
    if not os.path.exists(path):
        return None
    payload = np.load(path)
    points = payload["points"]
    lengths = payload["lengths"]
    series_by_conv = []
    cursor = 0
    for length in lengths.tolist():
        next_cursor = cursor + int(length)
        series_by_conv.append(np.asarray(points[cursor:next_cursor], dtype=float))
        cursor = next_cursor
    return series_by_conv


def collect_full_sync_series(ws: int, smooth_mode: str, granularity: str):
    series_by_conv = []
    started = time.time()
    for idx, conv_id in enumerate(conv_ids, start=1):
        conv_cache, _ = build_conversation_cache(conv_id, 'seeker', granularity, persist=False)
        if not conv_cache or not conv_cache.get('bg_utterances'):
            continue
        series = {
            dim: _compute_diff_series(dim, ws, smooth_mode, conv_cache)
            for dim in ['valence', 'arousal', 'dominance']
        }
        if any(series[dim] is None or len(series[dim]['prev']['y']) == 0 for dim in series):
            continue
        sizes = [len(series[dim]['prev']['y']) for dim in series]
        size = min(sizes)
        if size <= 0:
            continue
        stacked = np.column_stack([
            np.asarray(series['valence']['prev']['y'][:size], dtype=float),
            np.asarray(series['arousal']['prev']['y'][:size], dtype=float),
            np.asarray(series['dominance']['prev']['y'][:size], dtype=float),
        ])
        if len(stacked) == 0:
            continue
        series_by_conv.append(stacked)

        if idx % 100 == 0:
            elapsed = time.time() - started
            print(
                f"  scanned {idx}/{len(conv_ids)} convs | kept {len(series_by_conv)} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return series_by_conv


def build_sync_dataset_from_series(series_by_conv, tail_ratio: float, smooth_mode: str, granularity: str):
    dataset_points = []
    used_conversations = 0
    for stacked in series_by_conv:
        size = len(stacked)
        if size <= 0:
            continue
        if tail_ratio <= 0:
            start_idx = 0
        else:
            tail_count = max(1, int(np.ceil(size * tail_ratio)))
            start_idx = max(0, size - tail_count)
        tail_points = stacked[start_idx:]
        if len(tail_points) == 0:
            continue
        dataset_points.append(tail_points)
        used_conversations += 1

    if not dataset_points:
        return None

    points = np.vstack(dataset_points)
    alpha = (1.0 - SYNC_CONFIDENCE) / 2.0
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False) if len(points) > 1 else np.eye(3) * 1e-6
    cov = np.asarray(cov, dtype=float) + np.eye(3) * 1e-6
    result = {
        'sample_points': _sample_points(points),
        'mean': mean,
        'cov': cov,
        'chi2_threshold': float(chi2.ppf(SYNC_CONFIDENCE, df=3)),
        'low': np.quantile(points, alpha, axis=0),
        'high': np.quantile(points, 1.0 - alpha, axis=0),
        'used_conversations': used_conversations,
        'point_count': int(len(points)),
        'tail_ratio': tail_ratio,
        'confidence': SYNC_CONFIDENCE,
        'granularity': granularity,
        'smooth_mode': smooth_mode,
    }
    result['kde_data'] = _build_sync_kde_cache(result['sample_points'])
    return result


def save_sync_dataset(tail_ratio: float, ws: int, smooth_mode: str, granularity: str, dataset):
    cache_path = _sync_cache_path(tail_ratio, ws, smooth_mode, granularity)
    serializable = _serialize_sync_dataset(dataset)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return cache_path


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
        default="context",
        help="Comma-separated smooth modes. Default: context")
    parser.add_argument(
        "--granularities",
        default="sentence",
        help="Comma-separated granularities. Default: sentence")
    parser.add_argument(
        "--rebuild-series-cache",
        action="store_true",
        help="Ignore reusable full-scan cache and rescan conversations from scratch.")
    args = parser.parse_args()

    tail_pcts = parse_int_list(args.tail_pcts)
    window_sizes = parse_int_list(args.window_sizes)
    smooth_modes = parse_str_list(args.smooth_modes)
    granularities = parse_str_list(args.granularities)

    jobs = []
    grouped_jobs = []
    for ws in window_sizes:
        for smooth_mode in smooth_modes:
            for granularity in granularities:
                actual_mode = 'context' if smooth_mode == 'context' and granularity == 'sentence' else 'avg'
                grouped_jobs.append((ws, actual_mode, granularity))

    total_outputs = len(grouped_jobs) * len(tail_pcts)
    print(
        f"Preparing {len(grouped_jobs)} reusable scan group(s) -> {total_outputs} cache file(s)...",
        flush=True,
    )

    written = 0
    for group_idx, (ws, smooth_mode, granularity) in enumerate(grouped_jobs, start=1):
        group_started = time.time()
        series_cache_path = _series_cache_path(ws, smooth_mode, granularity)
        series_by_conv = None if args.rebuild_series_cache else load_series_cache(series_cache_path)
        if series_by_conv is None:
            print(
                f"[group {group_idx}/{len(grouped_jobs)}] scan once: ws={ws} mode={smooth_mode} gran={granularity}",
                flush=True,
            )
            series_by_conv = collect_full_sync_series(ws, smooth_mode, granularity)
            save_series_cache(series_cache_path, series_by_conv)
            scan_elapsed = time.time() - group_started
            print(
                f"[group {group_idx}/{len(grouped_jobs)}] scan done: kept {len(series_by_conv)} convs "
                f"time={scan_elapsed:.1f}s series_cache={series_cache_path}",
                flush=True,
            )
        else:
            scan_elapsed = time.time() - group_started
            print(
                f"[group {group_idx}/{len(grouped_jobs)}] reused full scan: kept {len(series_by_conv)} convs "
                f"time={scan_elapsed:.1f}s series_cache={series_cache_path}",
                flush=True,
            )

        for tail_pct in tail_pcts:
            tail_started = time.time()
            tail_ratio = tail_pct / 100.0
            dataset = build_sync_dataset_from_series(series_by_conv, tail_ratio, smooth_mode, granularity)
            if dataset is None:
                print(
                    f"  [tail {tail_pct}%] FAILED ws={ws} mode={smooth_mode} gran={granularity}",
                    flush=True,
                )
                continue
            cache_path = save_sync_dataset(tail_ratio, ws, smooth_mode, granularity, dataset)
            written += 1
            print(
                f"  [tail {tail_pct}%] OK points={dataset['point_count']} convs={dataset['used_conversations']} "
                f"time={time.time() - tail_started:.1f}s cache={cache_path}",
                flush=True,
            )

    print(f"Finished. Wrote {written}/{total_outputs} cache file(s).", flush=True)


if __name__ == "__main__":
    main()
