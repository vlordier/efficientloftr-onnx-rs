use std::cmp;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use efficientloftr_onnx_rs::{EfficientLoftrConfig, EfficientLoftrMatcher, GrayscaleFrame};

#[derive(Parser, Debug)]
#[command(name = "eval-video-frames")]
#[command(about = "Evaluate EfficientLoFTR ONNX matcher on sorted frame sequences")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    frames_dir: PathBuf,
    #[arg(long, default_value = "frame_")]
    frame_prefix: String,
    #[arg(long, default_value = "jpg")]
    frame_ext: String,
    #[arg(long)]
    output_csv: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    start_index: usize,
    #[arg(long)]
    end_index: Option<usize>,
    #[arg(long, default_value_t = 1)]
    step: usize,
    #[arg(long)]
    max_pairs: Option<usize>,
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
    if args.step == 0 {
        return Err("--step must be >= 1".to_string());
    }

    let mut frame_paths =
        collect_frame_paths(&args.frames_dir, &args.frame_prefix, &args.frame_ext)
            .map_err(|e| e.to_string())?;
    if frame_paths.len() < 2 {
        return Err(format!(
            "need at least 2 frames in {}",
            args.frames_dir.display()
        ));
    }

    let start = cmp::min(args.start_index, frame_paths.len() - 1);
    let end_exclusive = args
        .end_index
        .map(|v| cmp::min(v, frame_paths.len()))
        .unwrap_or(frame_paths.len());
    if end_exclusive <= start + 1 {
        return Err("selected frame range is too small for pair evaluation".to_string());
    }

    frame_paths = frame_paths[start..end_exclusive].to_vec();

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

    let mut csv_lines = vec!["pair,image0,image1,match_count".to_string()];
    let mut counts: Vec<usize> = Vec::new();
    let mut pair_id: usize = 0;

    let mut i = 0usize;
    while i + args.step < frame_paths.len() {
        if let Some(limit) = args.max_pairs
            && pair_id >= limit
        {
            break;
        }

        let frame0 = GrayscaleFrame::from_path(&frame_paths[i]).map_err(|e| e.to_string())?;
        let frame1 =
            GrayscaleFrame::from_path(&frame_paths[i + args.step]).map_err(|e| e.to_string())?;

        if frame0.width != frame1.width || frame0.height != frame1.height {
            return Err(format!(
                "frame dimensions differ: {} ({:?}x{:?}) vs {} ({:?}x{:?})",
                frame_paths[i].display(),
                frame0.width,
                frame0.height,
                frame_paths[i + args.step].display(),
                frame1.width,
                frame1.height
            ));
        }

        let matches = matcher
            .match_pair(&frame0, &frame1)
            .map_err(|e| format!("pair {pair_id} failed: {e}"))?;
        let count = matches.confidence.len();
        counts.push(count);
        csv_lines.push(format!(
            "{},{},{},{}",
            pair_id,
            frame_paths[i].display(),
            frame_paths[i + args.step].display(),
            count
        ));
        pair_id += 1;
        i += 1;
    }

    if counts.is_empty() {
        return Err("no frame pairs were evaluated".to_string());
    }

    if let Some(path) = args.output_csv {
        fs::write(path, csv_lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    }

    let mut sorted = counts.clone();
    sorted.sort_unstable();
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = counts.iter().copied().sum::<usize>() as f64 / counts.len() as f64;
    let p10 = percentile(&sorted, 0.10);
    let p50 = percentile(&sorted, 0.50);
    let p90 = percentile(&sorted, 0.90);

    println!("pairs: {}", counts.len());
    println!("mean matches: {:.2}", mean);
    println!("min/max: {}/{}", min, max);
    println!("p10/p50/p90: {}/{}/{}", p10, p50, p90);

    Ok(())
}

fn collect_frame_paths(
    frames_dir: &Path,
    frame_prefix: &str,
    frame_ext: &str,
) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut paths: Vec<PathBuf> = fs::read_dir(frames_dir)?
        .filter_map(|entry| entry.ok().map(|v| v.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| {
                    name.starts_with(frame_prefix)
                        && name
                            .to_ascii_lowercase()
                            .ends_with(&format!(".{}", frame_ext.to_ascii_lowercase()))
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn percentile(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let qq = q.clamp(0.0, 1.0);
    let idx = (qq * ((sorted.len() - 1) as f64)).round() as usize;
    sorted[idx]
}
