use std::fmt;
use std::path::Path;

use image::ImageReader;
use ndarray::{Array, ArrayViewD};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct EfficientLoftrConfig {
    pub input0_name: String,
    pub input1_name: String,
    pub keypoints0_name: String,
    pub keypoints1_name: String,
    pub confidence_name: String,
    pub confidence_threshold: f32,
    pub max_matches: usize,
}

impl Default for EfficientLoftrConfig {
    fn default() -> Self {
        Self {
            input0_name: "image0".to_string(),
            input1_name: "image1".to_string(),
            keypoints0_name: "keypoints0".to_string(),
            keypoints1_name: "keypoints1".to_string(),
            confidence_name: "confidence".to_string(),
            confidence_threshold: 0.2,
            max_matches: 512,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GrayscaleFrame {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl GrayscaleFrame {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, EfficientLoftrError> {
        let image = ImageReader::open(path)
            .map_err(|e| EfficientLoftrError::Io(e.to_string()))?
            .decode()
            .map_err(|e| EfficientLoftrError::Image(e.to_string()))?
            .to_luma8();
        Ok(Self {
            width: image.width() as usize,
            height: image.height() as usize,
            pixels: image.into_raw(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchOutput {
    pub keypoints0: Vec<[f32; 2]>,
    pub keypoints1: Vec<[f32; 2]>,
    pub confidence: Vec<f32>,
}

#[derive(Debug)]
pub enum EfficientLoftrError {
    Io(String),
    Image(String),
    InvalidImageShape,
    MissingInput(String),
    MissingOutput(String),
    InvalidOutputShape(String),
    Ort(String),
}

impl fmt::Display for EfficientLoftrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::Image(msg) => write!(f, "image decode error: {msg}"),
            Self::InvalidImageShape => write!(f, "invalid image shape or dimensions"),
            Self::MissingInput(name) => write!(f, "missing model input: {name}"),
            Self::MissingOutput(name) => write!(f, "missing model output: {name}"),
            Self::InvalidOutputShape(name) => write!(f, "invalid output shape for: {name}"),
            Self::Ort(msg) => write!(f, "onnx runtime error: {msg}"),
        }
    }
}

impl std::error::Error for EfficientLoftrError {}

pub struct EfficientLoftrMatcher {
    session: Session,
    config: EfficientLoftrConfig,
}

impl EfficientLoftrMatcher {
    pub fn from_model_path(
        model_path: impl AsRef<Path>,
        config: EfficientLoftrConfig,
    ) -> Result<Self, EfficientLoftrError> {
        let session = Session::builder()
            .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?;
        Ok(Self { session, config })
    }

    pub fn match_pair(
        &mut self,
        image0: &GrayscaleFrame,
        image1: &GrayscaleFrame,
    ) -> Result<MatchOutput, EfficientLoftrError> {
        if image0.width == 0
            || image0.height == 0
            || image1.width != image0.width
            || image1.height != image0.height
            || image0.pixels.len() != image0.width * image0.height
            || image1.pixels.len() != image1.width * image1.height
        {
            return Err(EfficientLoftrError::InvalidImageShape);
        }

        let input0 = make_nchw_tensor(image0)?;
        let input1 = make_nchw_tensor(image1)?;

        let available_input_names = self
            .session
            .inputs()
            .iter()
            .map(|entry| entry.name().to_string())
            .collect::<Vec<_>>();
        let input0_name = resolve_name(
            &available_input_names,
            &self.config.input0_name,
            &["image0", "img0", "input0", "left"],
        )
        .ok_or_else(|| {
            EfficientLoftrError::MissingInput(format!(
                "{} (available: {})",
                self.config.input0_name,
                available_input_names.join(", ")
            ))
        })?;
        let input1_name = resolve_name(
            &available_input_names,
            &self.config.input1_name,
            &["image1", "img1", "input1", "right"],
        )
        .ok_or_else(|| {
            EfficientLoftrError::MissingInput(format!(
                "{} (available: {})",
                self.config.input1_name,
                available_input_names.join(", ")
            ))
        })?;

        let available_output_names = self
            .session
            .outputs()
            .iter()
            .map(|entry| entry.name().to_string())
            .collect::<Vec<_>>();
        let keypoints0_name = resolve_name(
            &available_output_names,
            &self.config.keypoints0_name,
            &["keypoints0", "mkpts0_f", "points0", "kpts0"],
        )
        .ok_or_else(|| {
            EfficientLoftrError::MissingOutput(format!(
                "{} (available: {})",
                self.config.keypoints0_name,
                available_output_names.join(", ")
            ))
        })?;
        let keypoints1_name = resolve_name(
            &available_output_names,
            &self.config.keypoints1_name,
            &["keypoints1", "mkpts1_f", "points1", "kpts1"],
        )
        .ok_or_else(|| {
            EfficientLoftrError::MissingOutput(format!(
                "{} (available: {})",
                self.config.keypoints1_name,
                available_output_names.join(", ")
            ))
        })?;
        let confidence_name = resolve_name(
            &available_output_names,
            &self.config.confidence_name,
            &["confidence", "mconf", "scores", "conf"],
        )
        .ok_or_else(|| {
            EfficientLoftrError::MissingOutput(format!(
                "{} (available: {})",
                self.config.confidence_name,
                available_output_names.join(", ")
            ))
        })?;

        let mut outputs = self
            .session
            .run(ort::inputs![
                input0_name.as_str() => input0,
                input1_name.as_str() => input1
            ])
            .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?;

        let keypoints0_value = outputs.remove(keypoints0_name.as_str()).ok_or_else(|| {
            EfficientLoftrError::MissingOutput(format!(
                "{} (available: {})",
                keypoints0_name,
                available_output_names.join(", ")
            ))
        })?;
        let keypoints0 = keypoints0_value
            .try_extract_array::<f32>()
            .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?;
        let keypoints1_value = outputs.remove(keypoints1_name.as_str()).ok_or_else(|| {
            EfficientLoftrError::MissingOutput(format!(
                "{} (available: {})",
                keypoints1_name,
                available_output_names.join(", ")
            ))
        })?;
        let keypoints1 = keypoints1_value
            .try_extract_array::<f32>()
            .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?;
        let confidence_value = outputs.remove(confidence_name.as_str()).ok_or_else(|| {
            EfficientLoftrError::MissingOutput(format!(
                "{} (available: {})",
                confidence_name,
                available_output_names.join(", ")
            ))
        })?;
        let confidence = confidence_value
            .try_extract_array::<f32>()
            .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?;

        let k0 = decode_keypoints(keypoints0.view(), &self.config.keypoints0_name)?;
        let k1 = decode_keypoints(keypoints1.view(), &self.config.keypoints1_name)?;
        let conf = decode_confidence(confidence.view(), &self.config.confidence_name)?;

        let count = k0.len().min(k1.len()).min(conf.len());
        let mut out0 = Vec::with_capacity(count);
        let mut out1 = Vec::with_capacity(count);
        let mut outc = Vec::with_capacity(count);
        for i in 0..count {
            if conf[i] < self.config.confidence_threshold {
                continue;
            }
            out0.push(k0[i]);
            out1.push(k1[i]);
            outc.push(conf[i]);
            if outc.len() >= self.config.max_matches.max(1) {
                break;
            }
        }

        Ok(MatchOutput {
            keypoints0: out0,
            keypoints1: out1,
            confidence: outc,
        })
    }
}

fn make_nchw_tensor(image: &GrayscaleFrame) -> Result<Tensor<f32>, EfficientLoftrError> {
    let data: Vec<f32> = image.pixels.iter().map(|v| *v as f32 / 255.0).collect();
    let array = Array::from_shape_vec((1usize, 1usize, image.height, image.width), data)
        .map_err(|e| EfficientLoftrError::Ort(e.to_string()))?;
    Tensor::from_array(array).map_err(|e| EfficientLoftrError::Ort(e.to_string()))
}

fn resolve_name(available: &[String], preferred: &str, aliases: &[&str]) -> Option<String> {
    if available.iter().any(|name| name == preferred) {
        return Some(preferred.to_string());
    }
    aliases
        .iter()
        .find_map(|candidate| available.iter().find(|name| name.as_str() == *candidate))
        .cloned()
}

fn decode_keypoints(
    array: ArrayViewD<'_, f32>,
    name: &str,
) -> Result<Vec<[f32; 2]>, EfficientLoftrError> {
    match array.shape() {
        [n, c] if *c >= 2 => {
            let mut out = Vec::with_capacity(*n);
            for i in 0..*n {
                out.push([array[[i, 0]], array[[i, 1]]]);
            }
            Ok(out)
        }
        [b, n, c] if *b == 1 && *c >= 2 => {
            let mut out = Vec::with_capacity(*n);
            for i in 0..*n {
                out.push([array[[0, i, 0]], array[[0, i, 1]]]);
            }
            Ok(out)
        }
        _ => Err(EfficientLoftrError::InvalidOutputShape(name.to_string())),
    }
}

fn decode_confidence(
    array: ArrayViewD<'_, f32>,
    name: &str,
) -> Result<Vec<f32>, EfficientLoftrError> {
    match array.shape() {
        [n] => Ok((0..*n).map(|i| array[[i]]).collect()),
        [b, n] if *b == 1 => Ok((0..*n).map(|i| array[[0, i]]).collect()),
        _ => Err(EfficientLoftrError::InvalidOutputShape(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array;

    use super::{decode_confidence, decode_keypoints};

    #[test]
    fn decodes_supported_keypoint_shapes() {
        let k2 = Array::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).expect("shape");
        let k3 = Array::from_shape_vec((1, 2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).expect("shape");
        let out2 = decode_keypoints(k2.view().into_dyn(), "k2").expect("decode k2");
        let out3 = decode_keypoints(k3.view().into_dyn(), "k3").expect("decode k3");
        assert_eq!(out2, vec![[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(out3, vec![[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn decodes_supported_confidence_shapes() {
        let c1 = Array::from_shape_vec((3,), vec![0.1_f32, 0.2, 0.3]).expect("shape");
        let c2 = Array::from_shape_vec((1, 3), vec![0.1_f32, 0.2, 0.3]).expect("shape");
        let out1 = decode_confidence(c1.view().into_dyn(), "c1").expect("decode c1");
        let out2 = decode_confidence(c2.view().into_dyn(), "c2").expect("decode c2");
        assert_eq!(out1, vec![0.1, 0.2, 0.3]);
        assert_eq!(out2, vec![0.1, 0.2, 0.3]);
    }
}
