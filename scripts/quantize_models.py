#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate quantized EfficientLoFTR ONNX model variants."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source ONNX model")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated model variants",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=["fp16", "dynamic-qint8", "dynamic-quint8"],
        help="Quantization mode to generate (repeatable). Defaults to all modes.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON manifest path for generated variants",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import onnx
        from onnxconverter_common import float16
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit(
            "missing Python dependencies; install onnx, onnxruntime, and onnxconverter-common"
        ) from exc

    modes = args.mode or ["fp16", "dynamic-qint8", "dynamic-quint8"]
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    stem = input_path.stem

    for mode in modes:
        if mode == "fp16":
            output_path = output_dir / f"{stem}.fp16.onnx"
            model = float16.convert_float_to_float16_model_path(
                str(input_path),
                keep_io_types=True,
            )
            onnx.save(model, str(output_path))
        else:
            output_path = output_dir / f"{stem}.{mode}.onnx"
            weight_type = QuantType.QInt8 if mode == "dynamic-qint8" else QuantType.QUInt8
            quantize_dynamic(
                str(input_path),
                str(output_path),
                weight_type=weight_type,
                per_channel=True,
                extra_options={"MatMulConstBOnly": True},
            )

        manifest.append(
            {
                "mode": mode,
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
            }
        )
        print(f"generated {mode}: {output_path}")

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote manifest: {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())