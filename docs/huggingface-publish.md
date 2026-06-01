# Hugging Face Publish Recipe

This repository keeps the generated comparison artifacts local, but you can mirror them into your own Hugging Face repository with the `huggingface-cli` tools.

First, authenticate locally:

```bash
huggingface-cli login
```

Then upload the source weight and the compatibility/quantized artifacts you want to publish:

```bash
huggingface-cli upload <your-username>/<your-repo> samples/eloftr_640x480.onnx \
  --path-in-repo eloftr_640x480.onnx

huggingface-cli upload <your-username>/<your-repo> outputs/quantized_repair/eloftr_640x480.dynamic-qint8-full.compat.onnx \
  --path-in-repo quantized/eloftr_640x480.dynamic-qint8-full.compat.onnx

huggingface-cli upload <your-username>/<your-repo> outputs/quantized_repair/eloftr_640x480.dynamic-quint8-full.compat.onnx \
  --path-in-repo quantized/eloftr_640x480.dynamic-quint8-full.compat.onnx

huggingface-cli upload <your-username>/<your-repo> outputs/quantized_repair/eloftr_640x480.fp16-safe.compat.onnx \
  --path-in-repo quantized/eloftr_640x480.fp16-safe.compat.onnx

huggingface-cli upload <your-username>/<your-repo> outputs/quantized_repair/eloftr_640x480.fp16-full.compat.onnx \
  --path-in-repo quantized/eloftr_640x480.fp16-full.compat.onnx

huggingface-cli upload <your-username>/<your-repo> outputs/quantized_compare/relative_metrics.csv \
  --path-in-repo reports/relative_metrics.csv

huggingface-cli upload <your-username>/<your-repo> outputs/quantized_compare/relative_comparison.png \
  --path-in-repo reports/relative_comparison.png
```

Optional evaluation CSVs for traceability:

```bash
huggingface-cli upload <your-username>/<your-repo> outputs/quantized_repair/eval_summary.csv \
  --path-in-repo reports/eval_summary.csv

huggingface-cli upload <your-username>/<your-repo> outputs/quantized_compare/eval_summary_all.csv \
  --path-in-repo reports/eval_summary_all.csv
```

If you want the model card to mention the upstream source, include links to:

- https://zju3dv.github.io/efficientloftr/
- https://github.com/zju3dv/efficientloftr
- https://huggingface.co/zahilaty/EfficientLoFTR-ONNX
