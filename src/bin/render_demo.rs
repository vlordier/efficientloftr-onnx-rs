use std::cmp::Ordering;
use std::path::PathBuf;

use clap::Parser;
use efficientloftr_onnx_rs::{EfficientLoftrConfig, EfficientLoftrMatcher};
use image::imageops::FilterType;
use image::{GrayImage, ImageBuffer, Rgb, RgbImage};

#[derive(Parser, Debug)]
#[command(name = "render_demo")]
#[command(about = "Render EfficientLoFTR ONNX matches as a split-view image")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    image0: PathBuf,
    #[arg(long)]
    image1: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 640)]
    model_width: u32,
    #[arg(long, default_value_t = 480)]
    model_height: u32,
    #[arg(long)]
    viz_width: Option<u32>,
    #[arg(long)]
    viz_height: Option<u32>,
    #[arg(long, default_value_t = 800)]
    top_k: usize,
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
    #[arg(long, default_value_t = 0.0)]
    confidence_threshold: f32,
    #[arg(long, default_value_t = 4096)]
    max_matches: usize,
}

#[derive(Clone, Debug)]
struct MatchViz {
    p0: (f32, f32),
    p1: (f32, f32),
    conf: f32,
}

fn main() -> Result<(), String> {
    let args = Cli::parse();

    let viz_width = args.viz_width.unwrap_or(args.model_width);
    let viz_height = args.viz_height.unwrap_or(args.model_height);
    let split_x = (viz_width / 2).max(1);
    let right_panel_width = (viz_width - split_x).max(1);

    let left_rgb = load_rgb_resized(&args.image0, split_x, viz_height)?;
    let right_rgb = load_rgb_resized(&args.image1, right_panel_width, viz_height)?;
    let left_gray = load_grayscale_resized(&args.image0, args.model_width, args.model_height)?;
    let right_gray = load_grayscale_resized(&args.image1, args.model_width, args.model_height)?;

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

    let frame0 = efficientloftr_onnx_rs::GrayscaleFrame {
        width: args.model_width as usize,
        height: args.model_height as usize,
        pixels: left_gray.clone().into_raw(),
    };
    let frame1 = efficientloftr_onnx_rs::GrayscaleFrame {
        width: args.model_width as usize,
        height: args.model_height as usize,
        pixels: right_gray.clone().into_raw(),
    };

    let out = matcher
        .match_pair(&frame0, &frame1)
        .map_err(|e| e.to_string())?;

    let sx_left = split_x as f32 / args.model_width as f32;
    let sx_right = right_panel_width as f32 / args.model_width as f32;
    let sy = viz_height as f32 / args.model_height as f32;
    let mut matches: Vec<MatchViz> = out
        .confidence
        .iter()
        .enumerate()
        .map(|(i, conf)| MatchViz {
            p0: (out.keypoints0[i][0] * sx_left, out.keypoints0[i][1] * sy),
            p1: (
                split_x as f32 + out.keypoints1[i][0] * sx_right,
                out.keypoints1[i][1] * sy,
            ),
            conf: *conf,
        })
        .collect();

    matches.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(Ordering::Equal));
    if matches.len() > args.top_k {
        matches.truncate(args.top_k);
    }

    let canvas = render_matches(
        viz_width, viz_height, split_x, &left_rgb, &right_rgb, &matches,
    );
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    canvas.save(&args.output).map_err(|e| e.to_string())?;
    println!(
        "saved {} ({} matches)",
        args.output.display(),
        matches.len()
    );

    Ok(())
}

fn load_rgb_resized(path: &PathBuf, width: u32, height: u32) -> Result<RgbImage, String> {
    let img = image::open(path)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))?
        .to_rgb8();
    Ok(image::imageops::resize(
        &img,
        width,
        height,
        FilterType::Triangle,
    ))
}

fn load_grayscale_resized(path: &PathBuf, width: u32, height: u32) -> Result<GrayImage, String> {
    let img = image::open(path)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))?
        .to_luma8();
    Ok(image::imageops::resize(
        &img,
        width,
        height,
        FilterType::Triangle,
    ))
}

fn render_matches(
    canvas_width: u32,
    canvas_height: u32,
    split_x: u32,
    left: &RgbImage,
    right: &RgbImage,
    matches: &[MatchViz],
) -> RgbImage {
    let mut canvas: RgbImage = ImageBuffer::new(canvas_width, canvas_height);

    blit_rgb(&mut canvas, left, 0, 0);
    blit_rgb(&mut canvas, right, split_x, 0);

    let min_conf = matches.iter().map(|m| m.conf).fold(f32::INFINITY, f32::min);
    let max_conf = matches
        .iter()
        .map(|m| m.conf)
        .fold(f32::NEG_INFINITY, f32::max);
    let denom = (max_conf - min_conf).max(1e-6);

    for m in matches {
        let t = (m.conf - min_conf) / denom;
        let color = jet_color(t);
        let x0 = m.p0.0.round() as i32;
        let y0 = m.p0.1.round() as i32;
        let x1 = m.p1.0.round() as i32;
        let y1 = m.p1.1.round() as i32;
        draw_line(&mut canvas, x0, y0, x1, y1, color);
    }

    canvas
}

fn blit_rgb(canvas: &mut RgbImage, image: &RgbImage, ox: u32, oy: u32) {
    for y in 0..image.height() {
        for x in 0..image.width() {
            canvas.put_pixel(ox + x, oy + y, *image.get_pixel(x, y));
        }
    }
}

fn draw_line(canvas: &mut RgbImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && y0 >= 0 && (x0 as u32) < canvas.width() && (y0 as u32) < canvas.height() {
            canvas.put_pixel(x0 as u32, y0 as u32, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn jet_color(t: f32) -> Rgb<u8> {
    let x = t.clamp(0.0, 1.0);
    let r = (1.5 - (4.0 * x - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * x - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * x - 1.0).abs()).clamp(0.0, 1.0);
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}
