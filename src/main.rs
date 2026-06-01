use std::path::PathBuf;

use clap::Parser;
use efficientloftr_onnx_rs::{EfficientLoftrConfig, EfficientLoftrMatcher, GrayscaleFrame};

#[derive(Parser, Debug)]
#[command(name = "efficientloftr-onnx-rs")]
#[command(about = "Run EfficientLoFTR ONNX inference on a pair of grayscale images")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    image0: PathBuf,
    #[arg(long)]
    image1: PathBuf,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "image0")]
    input0_name: String,
    #[arg(long, default_value = "image1")]
    input1_name: String,
    #[arg(long, default_value = "keypoints0")]
    keypoints0_name: String,
    #[arg(long, default_value = "keypoints1")]
    keypoints1_name: String,
    #[arg(long, default_value = "confidence")]
    confidence_name: String,
    #[arg(long, default_value_t = 0.2)]
    confidence_threshold: f32,
    #[arg(long, default_value_t = 512)]
    max_matches: usize,
}

fn main() -> Result<(), String> {
    let args = Cli::parse();
    let frame0 = GrayscaleFrame::from_path(&args.image0).map_err(|e| e.to_string())?;
    let frame1 = GrayscaleFrame::from_path(&args.image1).map_err(|e| e.to_string())?;

    if frame0.width != frame1.width || frame0.height != frame1.height {
        return Err("image0 and image1 must have the same dimensions".to_string());
    }

    let config = EfficientLoftrConfig {
        input0_name: args.input0_name,
        input1_name: args.input1_name,
        keypoints0_name: args.keypoints0_name,
        keypoints1_name: args.keypoints1_name,
        confidence_name: args.confidence_name,
        confidence_threshold: args.confidence_threshold,
        max_matches: args.max_matches,
    };
    let mut matcher =
        EfficientLoftrMatcher::from_model_path(&args.model, config).map_err(|e| e.to_string())?;
    let matches = matcher
        .match_pair(&frame0, &frame1)
        .map_err(|e| e.to_string())?;

    if let Some(path) = args.output_json {
        let payload = serde_json::to_string_pretty(&matches).map_err(|e| e.to_string())?;
        std::fs::write(&path, payload).map_err(|e| e.to_string())?;
        println!("{}", path.display());
    } else {
        println!("matches: {}", matches.confidence.len());
    }

    Ok(())
}
