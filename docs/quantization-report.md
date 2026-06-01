# Quantization Report

This report captures the current quantization sweep for the validated sample model in this repository.

Source model:

- `samples/eloftr_640x480.onnx`
- size: `70,901,145` bytes

Generated variants:

- `outputs/quantized/eloftr_640x480.fp16.onnx` (`36,213,294` bytes)
- `outputs/quantized/eloftr_640x480.dynamic-qint8.onnx` (`41,900,574` bytes)
- `outputs/quantized/eloftr_640x480.dynamic-quint8.onnx` (`41,900,584` bytes)

Quantization command:

```bash
/path/to/python3 scripts/quantize_models.py \
  --input samples/eloftr_640x480.onnx \
  --output-dir outputs/quantized \
  --manifest outputs/quantized/manifest.json
```

Evaluation command:

```bash
cargo run --release --bin eval_video_frames -- \
  --model samples/eloftr_640x480.onnx \
  --model outputs/quantized/eloftr_640x480.fp16.onnx \
  --model outputs/quantized/eloftr_640x480.dynamic-qint8.onnx \
  --model outputs/quantized/eloftr_640x480.dynamic-quint8.onnx \
  --frames-dir samples/videos/scene0756_frames \
  --max-pairs 20 \
  --max-matches 4096 \
  --batch-size 1 \
  --summary-csv outputs/quantized/eval_summary.csv \
  --output-csv outputs/quantized/eval_pairs.csv
```

Results:

- Baseline float32 model executed successfully.
- Baseline summary: `pairs=20`, `mean=3627.30`, `min=752`, `max=4096`, `p10=1111`, `p50=4096`, `p90=4096`.
- `fp16` failed at model load time with a float16/float type mismatch inside `/matcher/loftr_coarse/layers.0/attention_1/Cast`.
- Dynamic `qint8` failed on the first pair with a reshape error in `/matcher/fine_matching/Reshape_2` after producing an empty `{0,64,64}` tensor.
- Dynamic `quint8` failed with the same reshape error as dynamic `qint8`.

Batching probe:

```bash
cargo run --release --bin eval_video_frames -- \
  --model samples/eloftr_640x480.onnx \
  --frames-dir samples/videos/scene0756_frames \
  --max-pairs 8 \
  --max-matches 4096 \
  --batch-size 4
```

- The Rust matcher now supports batched tensors, but this sample ONNX export is fixed to input batch `1` and rejects `batch-size 4` with `Got: 4 Expected: 1`.