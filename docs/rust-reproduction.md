# Rust Reproduction Guide

This repository is based on the upstream EfficientLoFTR project and uses the same sample weight source:

- [EfficientLoFTR project page](https://zju3dv.github.io/efficientloftr/)
- [EfficientLoFTR GitHub repository](https://github.com/zju3dv/efficientloftr)
- [Validated ONNX weights source](https://huggingface.co/zahilaty/EfficientLoFTR-ONNX)

The upstream repository includes reproduction entry points such as:

- `bash scripts/reproduce_test/outdoor_full_auc.sh`
- `bash scripts/reproduce_test/outdoor_opt_auc.sh`
- `bash scripts/reproduce_test/indoor_full_auc.sh`
- `bash scripts/reproduce_test/indoor_opt_auc.sh`

For this Rust port, the equivalent evaluation flow is:

```bash
cargo run --release --bin eval_video_frames -- \
  --model samples/eloftr_640x480.onnx \
  --frames-dir samples/videos/scene0756_frames \
  --max-pairs 20 \
  --max-matches 4096 \
  --summary-csv outputs/quantized_compare/eval_summary.csv \
  --output-csv outputs/quantized_compare/eval_pairs.csv
```

Quantized comparison sweep:

```bash
/path/to/python3 scripts/quantize_models.py \
  --input samples/eloftr_640x480.onnx \
  --output-dir outputs/quantized_repair \
  --manifest outputs/quantized_repair/manifest.json \
  --mode dynamic-qint8-full \
  --mode dynamic-quint8-full \
  --mode fp16-safe \
  --mode fp16-full

cargo run --release --bin eval_video_frames -- \
  --model samples/eloftr_640x480.onnx \
  --model outputs/quantized_repair/eloftr_640x480.dynamic-qint8-full.compat.onnx \
  --model outputs/quantized_repair/eloftr_640x480.dynamic-quint8-full.compat.onnx \
  --model outputs/quantized_repair/eloftr_640x480.fp16-safe.compat.onnx \
  --model outputs/quantized_repair/eloftr_640x480.fp16-full.compat.onnx \
  --frames-dir samples/videos/scene0756_frames \
  --max-pairs 5 \
  --max-matches 4096 \
  --summary-csv outputs/quantized_repair/eval_summary.csv \
  --output-csv outputs/quantized_repair/eval_pairs.csv
```

Relative comparison artifacts are generated with:

```bash
/path/to/python3 outputs/quantized_compare/make_relative_plot.py
```

That produces:

- `outputs/quantized_compare/eval_summary_all.csv`
- `outputs/quantized_compare/relative_metrics.csv`
- `outputs/quantized_compare/relative_comparison.png`

If you want to mirror the weights and comparison artifacts into your own Hugging Face repository, upload:

- the original ONNX weight from the source repo
- the compatibility/runnable quantized ONNX variants
- `relative_metrics.csv`
- `relative_comparison.png`
- the evaluation CSVs for traceability

The exact repository name and upload method depend on your Hugging Face target and auth token, so this repo keeps the artifacts local and reproducible.