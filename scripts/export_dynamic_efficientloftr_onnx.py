#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from upstream_efficientloftr import BatchedTopKWrapper, load_upstream_matcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a true dynamic-batch ONNX model from upstream EfficientLoFTR PyTorch weights."
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("third_party/EfficientLoFTR"),
        help="Path to upstream EfficientLoFTR repository checkout",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("third_party/weights/weights/eloftr_outdoor.ckpt"),
        help="Path to upstream EfficientLoFTR checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--model-type",
        choices=["full", "opt"],
        default="full",
        help="Upstream config variant",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="ONNX output path",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--example-batch", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-matches", type=int, default=4096)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--no-dynamo",
        action="store_true",
        help="Use legacy tracer exporter instead of dynamo exporter",
    )
    parser.add_argument(
        "--export-safe",
        action="store_true",
        help="Use export-safe fine matching path (skips fragile local refinement branch)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.example_batch < 1:
        raise SystemExit("--example-batch must be >= 1")
    if args.max_batch < args.example_batch:
        raise SystemExit("--max-batch must be >= --example-batch")

    matcher = load_upstream_matcher(
        upstream_root=args.upstream_root,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        export_safe=args.export_safe,
    )
    wrapper = BatchedTopKWrapper(
        matcher,
        max_matches=args.max_matches,
        max_batch=args.max_batch,
    ).eval()

    x0 = torch.rand(args.example_batch, 1, args.height, args.width, dtype=torch.float32)
    x1 = torch.rand(args.example_batch, 1, args.height, args.width, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["image0", "image1"]
    output_names = ["keypoints0", "keypoints1", "confidence"]

    if args.no_dynamo:
        torch.onnx.export(
            wrapper,
            (x0, x1),
            f=str(args.output),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
            dynamic_axes={
                "image0": {0: "batch"},
                "image1": {0: "batch"},
                "keypoints0": {0: "batch"},
                "keypoints1": {0: "batch"},
                "confidence": {0: "batch"},
            },
        )
    else:
        batch_dim = torch.export.Dim("batch", min=1, max=args.max_batch)
        torch.onnx.export(
            wrapper,
            (x0, x1),
            f=str(args.output),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            dynamo=True,
            dynamic_shapes={
                "image0": {0: batch_dim},
                "image1": {0: batch_dim},
            },
        )

    model = onnx.load(str(args.output))
    onnx.checker.check_model(model)
    print(f"exported: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())