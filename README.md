# efficientloftr-onnx-rs

Rust ONNX Runtime wrapper for EfficientLoFTR-style semi-dense matching.

## What this provides

- Native Rust library API for running ONNX inference on image pairs
- CLI for quick matching experiments
- Flexible output tensor names for different exported model variants

## Build

```bash
cargo build --release
```

## CLI usage

```bash
cargo run --release -- \
  --model /path/to/efficient_loftr.onnx \
  --image0 /path/to/image0.png \
  --image1 /path/to/image1.png \
  --output-json /tmp/matches.json
```

Optional tensor names:

- `--input0-name` (default: `image0`)
- `--input1-name` (default: `image1`)
- `--keypoints0-name` (default: `keypoints0`)
- `--keypoints1-name` (default: `keypoints1`)
- `--confidence-name` (default: `confidence`)

## Library sketch

```rust
use efficientloftr_onnx_rs::{EfficientLoftrConfig, EfficientLoftrMatcher, GrayscaleFrame};

let cfg = EfficientLoftrConfig::default();
let mut matcher = EfficientLoftrMatcher::from_model_path("model.onnx", cfg)?;
let image0 = GrayscaleFrame::from_path("a.png")?;
let image1 = GrayscaleFrame::from_path("b.png")?;
let matches = matcher.match_pair(&image0, &image1)?;
println!("{} matches", matches.confidence.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```
