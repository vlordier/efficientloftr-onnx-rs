use std::cmp;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use efficientloftr_onnx_rs::{EfficientLoftrConfig, EfficientLoftrMatcher, GrayscaleFrame};

#[derive(Parser, Debug)]
#[command(name = "eval-video-frames")]
#[command(about = "Evaluate EfficientLoFTR ONNX matcher on sorted frame sequences")]
struct Cli {
    #[arg(long, required = true)]
    model: Vec<PathBuf>,
    #[arg(long)]
    frames_dir: PathBuf,
    #[arg(long, default_value = "frame_")]
    frame_prefix: String,
    #[arg(long, default_value = "jpg")]
    frame_ext: String,
    #[arg(long)]
    output_csv: Option<PathBuf>,
    #[arg(long)]
    summary_csv: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    start_index: usize,
    #[arg(long)]
    end_index: Option<usize>,
    #[arg(long, default_value_t = 1)]
    step: usize,
    #[arg(long)]
    max_pairs: Option<usize>,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
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
    if args.batch_size == 0 {
        return Err("--batch-size must be >= 1".to_string());
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

    let pair_specs = collect_pair_specs(&frame_paths, args.step, args.max_pairs);
    if pair_specs.is_empty() {
        return Err("no frame pairs were evaluated".to_string());
    }

    let mut pair_csv_lines = vec!["model,pair,image0,image1,match_count".to_string()];
    let mut summary_csv_lines = vec![
        "model,status,pairs,mean_matches,min_matches,max_matches,p10,p50,p90,batch_size,elapsed_ms,error"
            .to_string(),
    ];

    for model in &args.model {
        match evaluate_model(model, &config, &pair_specs, args.batch_size) {
            Ok(summary) => {
                for (pair_id, image0, image1, count) in &summary.rows {
                    pair_csv_lines.push(format!(
                        "{},{},{},{},{}",
                        model.display(),
                        pair_id,
                        image0.display(),
                        image1.display(),
                        count
                    ));
                }
                summary_csv_lines.push(format!(
                    "{},ok,{},{:.2},{},{},{},{},{},{},{},",
                    model.display(),
                    summary.counts.len(),
                    summary.mean,
                    summary.min,
                    summary.max,
                    summary.p10,
                    summary.p50,
                    summary.p90,
                    args.batch_size,
                    summary.elapsed_ms
                ));

                println!("model: {}", model.display());
                println!("status: ok");
                println!("pairs: {}", summary.counts.len());
                println!("mean matches: {:.2}", summary.mean);
                println!("min/max: {}/{}", summary.min, summary.max);
                println!(
                    "p10/p50/p90: {}/{}/{}",
                    summary.p10, summary.p50, summary.p90
                );
                println!("elapsed ms: {}", summary.elapsed_ms);
            }
            Err(error) => {
                if args.model.len() == 1 {
                    return Err(error);
                }
                summary_csv_lines.push(format!(
                    "{},error,,,,,,,,,,{}",
                    model.display(),
                    csv_escape(&error)
                ));
                println!("model: {}", model.display());
                println!("status: error");
                println!("error: {}", error);
            }
        }
    }

    if let Some(path) = args.output_csv {
        fs::write(path, pair_csv_lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    }

    if let Some(path) = args.summary_csv {
        fs::write(path, summary_csv_lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    }

    Ok(())
}

struct EvalSummary {
    counts: Vec<usize>,
    rows: Vec<(usize, PathBuf, PathBuf, usize)>,
    mean: f64,
    min: usize,
    max: usize,
    p10: usize,
    p50: usize,
    p90: usize,
    elapsed_ms: u128,
}

fn collect_pair_specs(
    frame_paths: &[PathBuf],
    step: usize,
    max_pairs: Option<usize>,
) -> Vec<(usize, PathBuf, PathBuf)> {
    let mut pair_specs = Vec::new();
    let mut pair_id = 0usize;
    let mut i = 0usize;
    while i + step < frame_paths.len() {
        if let Some(limit) = max_pairs
            && pair_id >= limit
        {
            break;
        }
        pair_specs.push((
            pair_id,
            frame_paths[i].clone(),
            frame_paths[i + step].clone(),
        ));
        pair_id += 1;
        i += 1;
    }
    pair_specs
}

fn evaluate_model(
    model: &Path,
    config: &EfficientLoftrConfig,
    pair_specs: &[(usize, PathBuf, PathBuf)],
    batch_size: usize,
) -> Result<EvalSummary, String> {
    let mut matcher =
        EfficientLoftrMatcher::from_model_path(model, config.clone()).map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let mut rows = Vec::with_capacity(pair_specs.len());
    let mut counts = Vec::with_capacity(pair_specs.len());

    let mut offset = 0usize;
    while offset < pair_specs.len() {
        let upper = cmp::min(offset + batch_size, pair_specs.len());
        let batch_specs = &pair_specs[offset..upper];

        let mut frames0 = Vec::with_capacity(batch_specs.len());
        let mut frames1 = Vec::with_capacity(batch_specs.len());
        for (_, image0, image1) in batch_specs {
            let frame0 = GrayscaleFrame::from_path(image0).map_err(|e| e.to_string())?;
            let frame1 = GrayscaleFrame::from_path(image1).map_err(|e| e.to_string())?;
            if frame0.width != frame1.width || frame0.height != frame1.height {
                return Err(format!(
                    "frame dimensions differ: {} ({}x{}) vs {} ({}x{})",
                    image0.display(),
                    frame0.width,
                    frame0.height,
                    image1.display(),
                    frame1.width,
                    frame1.height
                ));
            }
            frames0.push(frame0);
            frames1.push(frame1);
        }

        let batch0: Vec<&GrayscaleFrame> = frames0.iter().collect();
        let batch1: Vec<&GrayscaleFrame> = frames1.iter().collect();
        let matches = matcher.match_batch(&batch0, &batch1).map_err(|e| {
            format!(
                "model {} batch starting at pair {} failed: {e}",
                model.display(),
                batch_specs[0].0
            )
        })?;

        for ((pair_id, image0, image1), match_output) in batch_specs.iter().zip(matches) {
            let count = match_output.confidence.len();
            counts.push(count);
            rows.push((*pair_id, image0.clone(), image1.clone(), count));
        }

        offset = upper;
    }

    if counts.is_empty() {
        return Err(format!(
            "model {} produced no evaluated pairs",
            model.display()
        ));
    }

    let mut sorted = counts.clone();
    sorted.sort_unstable();
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = counts.iter().copied().sum::<usize>() as f64 / counts.len() as f64;
    let p10 = percentile(&sorted, 0.10);
    let p50 = percentile(&sorted, 0.50);
    let p90 = percentile(&sorted, 0.90);

    Ok(EvalSummary {
        counts,
        rows,
        mean,
        min,
        max,
        p10,
        p50,
        p90,
        elapsed_ms: started.elapsed().as_millis(),
    })
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

fn csv_escape(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "'"))
}
