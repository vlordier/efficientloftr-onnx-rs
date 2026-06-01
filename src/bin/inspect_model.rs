use std::path::PathBuf;

use clap::Parser;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::ValueType;

#[derive(Parser, Debug)]
#[command(name = "inspect-model")]
#[command(about = "Inspect ONNX model inputs/outputs and batch support")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
}

fn main() -> Result<(), String> {
    let args = Cli::parse();
    let session = Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .commit_from_file(&args.model)
        .map_err(|e| e.to_string())?;

    println!("model: {}", args.model.display());
    println!("inputs:");
    for input in session.inputs() {
        println!(
            "  - {}: {}",
            input.name(),
            describe_value_type(input.dtype())
        );
    }
    println!("outputs:");
    for output in session.outputs() {
        println!(
            "  - {}: {}",
            output.name(),
            describe_value_type(output.dtype())
        );
    }
    println!(
        "batch-support: {}",
        classify_batch_support(session.inputs())
    );

    Ok(())
}

fn classify_batch_support(inputs: &[ort::value::Outlet]) -> &'static str {
    if inputs.is_empty() {
        return "unknown";
    }

    let mut saw_tensor = false;
    let mut all_dynamic = true;
    for input in inputs {
        if let ValueType::Tensor { shape, .. } = input.dtype() {
            saw_tensor = true;
            match shape.first().copied() {
                Some(-1) => {}
                Some(1) => all_dynamic = false,
                Some(_) => return "fixed-nonunit-batch",
                None => return "not-batched-tensor",
            }
        }
    }

    if !saw_tensor {
        "unknown"
    } else if all_dynamic {
        "dynamic-batch"
    } else {
        "fixed-batch-1"
    }
}

fn describe_value_type(value_type: &ValueType) -> String {
    match value_type {
        ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => {
            let dims = shape
                .iter()
                .zip(dimension_symbols.iter())
                .map(|(dim, symbol)| match (*dim, symbol.is_empty()) {
                    (-1, false) => format!("{symbol}=-1"),
                    (-1, true) => "-1".to_string(),
                    (value, false) => format!("{symbol}={value}"),
                    (value, true) => value.to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("tensor<{ty}>[{dims}]")
        }
        ValueType::Sequence(inner) => format!("sequence<{}>", describe_value_type(inner)),
        ValueType::Map { key, value } => format!("map<{key}, {value}>"),
        ValueType::Optional(inner) => format!("optional<{}>", describe_value_type(inner)),
    }
}
