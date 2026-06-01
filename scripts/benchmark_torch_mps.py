#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from upstream_efficientloftr import BatchedTopKWrapper, load_upstream_matcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark upstream EfficientLoFTR with torch MPS/CPU on frame pairs."
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("third_party/EfficientLoFTR"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("third_party/weights/weights/eloftr_outdoor.ckpt"),
    )
    parser.add_argument("--model-type", choices=["full", "opt"], default="full")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-prefix", default="frame_")
    parser.add_argument("--frame-ext", default="jpg")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-matches", type=int, default=4096)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    return parser.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested, but MPS is unavailable")
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_frames(frames_dir: Path, prefix: str, ext: str) -> list[Path]:
    ext_l = ext.lower()
    frames = [
        p
        for p in frames_dir.iterdir()
        if p.is_file()
        and p.name.startswith(prefix)
        and p.suffix.lower() == f".{ext_l}"
    ]
    frames.sort()
    return frames


def to_gray_tensor(path: Path, device: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = round(max(0.0, min(1.0, q)) * (len(values) - 1))
    return float(sorted(values)[idx])


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_batch < args.batch_size:
        raise SystemExit("--max-batch must be >= --batch-size")
    if args.step < 1:
        raise SystemExit("--step must be >= 1")

    device = pick_device(args.device)
    print(f"device: {device}")

    matcher = load_upstream_matcher(args.upstream_root, args.checkpoint, args.model_type)
    wrapper = BatchedTopKWrapper(
        matcher,
        max_matches=args.max_matches,
        max_batch=args.max_batch,
    ).to(device)
    wrapper.eval()

    frames = collect_frames(args.frames_dir, args.frame_prefix, args.frame_ext)
    if len(frames) < 2:
        raise SystemExit("need at least two frames")

    start = min(args.start_index, len(frames) - 1)
    end = min(args.end_index, len(frames)) if args.end_index is not None else len(frames)
    if end <= start + 1:
        raise SystemExit("selected frame range is too small for pair evaluation")
    frames = frames[start:end]

    pairs: list[tuple[int, Path, Path]] = []
    i = 0
    pair_id = 0
    while i + args.step < len(frames):
        if pair_id >= args.max_pairs:
            break
        pairs.append((pair_id, frames[i], frames[i + args.step]))
        i += 1
        pair_id += 1
    if not pairs:
        raise SystemExit("no frame pairs to evaluate")

    rows: list[dict[str, object]] = []
    counts: list[int] = []
    pair_latency_ms: list[float] = []
    batch_offset = 0

    started = time.perf_counter()
    with torch.no_grad():
        while batch_offset < len(pairs):
            upper = min(batch_offset + args.batch_size, len(pairs))
            batch_specs = pairs[batch_offset:upper]

            batch0 = torch.cat([to_gray_tensor(p0, device) for _, p0, _ in batch_specs], dim=0)
            batch1 = torch.cat([to_gray_tensor(p1, device) for _, _, p1 in batch_specs], dim=0)

            t0 = time.perf_counter()
            _, _, conf = wrapper(batch0, batch1)
            if device == "mps":
                torch.mps.synchronize()
            batch_ms = (time.perf_counter() - t0) * 1000.0
            per_pair_ms = batch_ms / len(batch_specs)

            valid = conf > 0
            for (pid, p0, p1), match_mask in zip(batch_specs, valid):
                match_count = int(match_mask.sum().item())
                counts.append(match_count)
                pair_latency_ms.append(per_pair_ms)
                rows.append(
                    {
                        "pair": pid,
                        "image0": str(p0),
                        "image1": str(p1),
                        "match_count": match_count,
                        "batch_latency_ms": batch_ms,
                        "per_pair_latency_ms": per_pair_ms,
                    }
                )

            batch_offset = upper

    elapsed_s = time.perf_counter() - started
    mean_matches = statistics.fmean(counts)
    p10 = int(percentile([float(c) for c in counts], 0.10))
    p50 = int(percentile([float(c) for c in counts], 0.50))
    p90 = int(percentile([float(c) for c in counts], 0.90))
    mean_latency = statistics.fmean(pair_latency_ms)
    p50_latency = percentile(pair_latency_ms, 0.50)
    pairs_per_sec = len(counts) / elapsed_s if elapsed_s > 0 else 0.0

    print("status: ok")
    print(f"pairs: {len(counts)}")
    print(f"mean matches: {mean_matches:.2f}")
    print(f"min/max: {min(counts)}/{max(counts)}")
    print(f"p10/p50/p90: {p10}/{p50}/{p90}")
    print(f"requested/effective batch size: {args.batch_size}/{args.batch_size}")
    print(f"elapsed ms: {int(elapsed_s * 1000.0)}")
    print(f"mean/p50 pair latency ms: {mean_latency:.3f}/{p50_latency:.3f}")
    print(f"pairs/sec: {pairs_per_sec:.3f}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pair",
                    "image0",
                    "image1",
                    "match_count",
                    "batch_latency_ms",
                    "per_pair_latency_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "backend",
                    "device",
                    "pairs",
                    "mean_matches",
                    "min_matches",
                    "max_matches",
                    "p10",
                    "p50",
                    "p90",
                    "requested_batch_size",
                    "effective_batch_size",
                    "elapsed_ms",
                    "mean_pair_latency_ms",
                    "p50_pair_latency_ms",
                    "pairs_per_sec",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "backend": "torch",
                    "device": device,
                    "pairs": len(counts),
                    "mean_matches": f"{mean_matches:.2f}",
                    "min_matches": min(counts),
                    "max_matches": max(counts),
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "requested_batch_size": args.batch_size,
                    "effective_batch_size": args.batch_size,
                    "elapsed_ms": int(elapsed_s * 1000.0),
                    "mean_pair_latency_ms": f"{mean_latency:.3f}",
                    "p50_pair_latency_ms": f"{p50_latency:.3f}",
                    "pairs_per_sec": f"{pairs_per_sec:.3f}",
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())