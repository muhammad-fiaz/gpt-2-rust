//! GPT-2 Rust 🦀🔥 — Unified CLI entry point.
//!
//! Project: https://github.com/muhammad-fiaz/gpt-2-rust
//! Author: Muhammad Fiaz (contact@muhammadfiaz.com)
//! Year: 2026
//! License: MIT License
//!
//! Exposes download, inference/generate, evaluate, and training modes under
//! a single, highly integrated CLI binary.

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Cuda};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::prelude::Module;
use burn::record::{CompactRecorder, Recorder};
use clap::Parser;
use std::path::Path;
use std::process::Command;

use gpt_2_rust::{
    generate as run_autoregressive_generation, loader, training, GenerationConfig, Gpt2Config,
    Gpt2Model, Gpt2Tokenizer,
};

#[derive(Parser, Debug)]
#[command(
    name    = "gpt2",
    version = "0.1.0",
    author  = "Muhammad Fiaz <contact@muhammadfiaz.com>",
    about   = "GPT-2 unified CLI — download weights, run inference, evaluate, or train (native Rust / Burn)",
)]
struct Args {
    // === Mode Flags ===
    /// Run model parameter downloader.
    #[arg(long)]
    download: bool,

    /// Run text generation / inference.
    #[arg(long, aliases = &["inference"])]
    generate: bool,

    /// Run pre-training / fine-tuning.
    #[arg(long)]
    train: bool,

    /// Run evaluation (perplexity & loss).
    #[arg(long, aliases = &["eval"])]
    evaluate: bool,

    // === Shared Configuration Parameters ===
    /// GPT-2 variant to download or model size: 'small', 'medium', 'large', 'xl', or 'all'.
    #[arg(long, default_value = "small")]
    size: String,

    /// Target weights output/input directory.
    #[arg(long, default_value = "weights")]
    weights_dir: String,

    /// Path to model weights file (safetensors or compact Burn format).
    #[arg(short, long)]
    model: Option<String>,

    /// Model format: 'safetensors' or 'compact' (Burn record format).
    #[arg(long, default_value = "safetensors")]
    format: String,

    /// Prompt text to condition text generation on.
    #[arg(short, long, default_value = "The future of artificial intelligence is")]
    prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value = "100")]
    max_new_tokens: usize,

    /// Sampling temperature (0 = greedy, 1.0 = unchanged).
    #[arg(long, default_value = "0.8")]
    temperature: f64,

    /// Top-K sampling cutoff (0 disables top-k).
    #[arg(long, default_value = "50")]
    top_k: usize,

    /// Top-P (nucleus) sampling cutoff (0.0 disables top-p).
    #[arg(long, default_value = "0.0")]
    top_p: f64,

    /// Stop generation when the EOT token is encountered.
    #[arg(long, default_value = "true")]
    stop_on_eot: bool,

    /// Force overwrite of existing files (downloader mode).
    #[arg(long, default_value = "false")]
    force: bool,

    /// Path to the text file (train or eval mode).
    #[arg(short, long, default_value = "data/input.txt")]
    data: String,

    /// Directory for checkpoints, logs, and final model (train mode).
    #[arg(long, default_value = "artifacts")]
    artifact_dir: String,

    /// Number of training epochs.
    #[arg(long, default_value = "3")]
    epochs: usize,

    /// Batch size.
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Sequence length (context window).
    #[arg(long, default_value = "128")]
    seq_len: usize,

    /// Peak learning rate (train mode).
    #[arg(long, default_value = "3e-4")]
    lr: f64,

    /// Fraction of data to use for validation (train mode).
    #[arg(long, default_value = "0.1")]
    val_fraction: f64,

    /// Dropout probability.
    #[arg(long, default_value = "0.1")]
    dropout: f64,

    /// Number of DataLoader worker threads.
    #[arg(long, default_value = "2")]
    workers: usize,

    /// Random seed for reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,
}

// === Downloader Runner ===

struct ModelFiles {
    variant_name: &'static str,
    repo_id: &'static str,
}

const VARIANTS: &[ModelFiles] = &[
    ModelFiles {
        variant_name: "small",
        repo_id: "openai-community/gpt2",
    },
    ModelFiles {
        variant_name: "medium",
        repo_id: "openai-community/gpt2-medium",
    },
    ModelFiles {
        variant_name: "large",
        repo_id: "openai-community/gpt2-large",
    },
    ModelFiles {
        variant_name: "xl",
        repo_id: "openai-community/gpt2-xl",
    },
];

fn download_file(url: &str, output_path: &Path, force: bool) -> Result<()> {
    if output_path.exists() && !force {
        log::info!("File already exists: {}. Skipping.", output_path.display());
        return Ok(());
    }

    log::info!("Downloading {} -> {}...", url, output_path.display());

    let status = Command::new("curl")
        .arg("-L")
        .arg("-o")
        .arg(output_path)
        .arg(url)
        .status()
        .context("Failed to run curl. Ensure 'curl' is installed and in PATH.")?;

    if !status.success() {
        anyhow::bail!("curl failed to download {}", url);
    }
    Ok(())
}

fn run_download(args: &Args) -> Result<()> {
    let base_dir = Path::new(&args.weights_dir);

    let selected_variants: Vec<&ModelFiles> = match args.size.as_str() {
        "all" => VARIANTS.iter().collect(),
        other => {
            if let Some(v) = VARIANTS.iter().find(|v| v.variant_name == other) {
                vec![v]
            } else {
                anyhow::bail!("Unknown variant '{}'. Choose: small, medium, large, xl, all", other);
            }
        }
    };

    log::info!("\x1b[1;36m=== Downloading GPT-2 Parameters ===\x1b[0m");
    log::info!("Variant(s)          : \x1b[1;33m{}\x1b[0m", args.size);
    log::info!("Output directory    : \x1b[1;32m{}\x1b[0m", base_dir.display());
    log::info!("\x1b[1;36m====================================\x1b[0m");

    for variant in selected_variants {
        let variant_dir = base_dir.join(variant.variant_name);
        let files = ["model.safetensors", "config.json", "vocab.json", "merges.txt"];
        
        let all_exist = !args.force && files.iter().all(|file| {
            variant_dir.join(file).exists()
        });

        if all_exist {
            log::info!(
                "\x1b[1;32mModel and configs all already downloaded and in place. Location: {}\x1b[0m",
                variant_dir.display().to_string().replace("\\", "/")
            );
            continue;
        }

        std::fs::create_dir_all(&variant_dir)?;
        for file in &files {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                variant.repo_id, file
            );
            let dest = variant_dir.join(file);
            download_file(&url, &dest, args.force)?;
        }
        log::info!("\x1b[1;32mCompleted downloads for variant '{}'.\x1b[0m", variant.variant_name);
    }
    Ok(())
}

// === Model Integrity Verification ===

fn verify_model_files(
    size: &str,
    model_path: &str,
) -> Result<()> {
    let mut missing_files = Vec::new();

    // Check weights file
    let weights_path = Path::new(model_path);
    if !weights_path.exists() {
        missing_files.push(model_path.replace("\\", "/"));
    }

    // Check config, vocab, and merges in the same folder as the weights file
    let parent_dir = weights_path.parent().unwrap_or_else(|| Path::new("."));
    
    let config_file = parent_dir.join("config.json");
    if !config_file.exists() {
        missing_files.push(config_file.to_string_lossy().replace("\\", "/"));
    }

    let vocab_file = parent_dir.join("vocab.json");
    if !vocab_file.exists() {
        missing_files.push(vocab_file.to_string_lossy().replace("\\", "/"));
    }

    let merges_file = parent_dir.join("merges.txt");
    if !merges_file.exists() {
        missing_files.push(merges_file.to_string_lossy().replace("\\", "/"));
    }

    if !missing_files.is_empty() {
        log::error!("\x1b[1;31m⚠️  WARNING: GPT-2 Model is not fully downloaded or files are missing! ⚠️\x1b[0m");
        log::error!("\x1b[1;33mThe following required file(s) are missing:\x1b[0m");
        for file in &missing_files {
            log::error!("  \x1b[1;31m- {}\x1b[0m", file);
        }
        
        log::error!("\x1b[1;32m💡 HELPER: To download the missing model weights and configuration files, run:\x1b[0m");
        log::error!("  \x1b[1;36mcargo run --release -- --download --size {}\x1b[0m\n", size);
        anyhow::bail!("Missing model files: {}", missing_files.join(", "));
    }

    Ok(())
}

// === Generator Runner ===

fn run_generate(args: &Args) -> Result<()> {
    type Backend = Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::default();

    let model_path = args.model.clone().unwrap_or_else(|| {
        format!("{}/{}/model.safetensors", args.weights_dir, args.size)
    });

    verify_model_files(&args.size, &model_path)?;

    let cfg = match args.size.as_str() {
        "small"  => Gpt2Config::gpt2_small(),
        "medium" => Gpt2Config::gpt2_medium(),
        "large"  => Gpt2Config::gpt2_large(),
        "xl"     => Gpt2Config::gpt2_xl(),
        other    => anyhow::bail!("Unknown model size '{}'", other),
    }
    .with_dropout(0.0);

    let mut model: Gpt2Model<Backend> = cfg.init(&device);

    log::info!("Loading weights from '{}'...", model_path);
    let tensor_map = loader::load_safetensors(&model_path)?;
    model = loader::load_gpt2_weights(model, &tensor_map, &device)?;

    let tokenizer = Gpt2Tokenizer::new()?;
    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        temperature:    args.temperature,
        top_k:          if args.top_k > 0 { Some(args.top_k) } else { None },
        top_p:          if args.top_p > 0.0 { Some(args.top_p) } else { None },
        seed:           args.seed,
        stop_on_eot:    args.stop_on_eot,
    };

    print!("{}", args.prompt);
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let _output = run_autoregressive_generation::<Backend>(&model, &tokenizer, &args.prompt, &gen_cfg, &device);

    std::mem::drop(model);
    println!();
    Ok(())
}

// === Evaluator Runner ===

fn run_eval(args: &Args) -> Result<()> {
    type Backend = Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::default();

    let model_path = args.model.clone().unwrap_or_else(|| {
        let ext = if args.format == "compact" { "mpk" } else { "safetensors" };
        format!("{}/{}/model.{}", args.weights_dir, args.size, ext)
    });

    verify_model_files(&args.size, &model_path)?;

    let cfg = match args.size.as_str() {
        "small"  => Gpt2Config::gpt2_small(),
        "medium" => Gpt2Config::gpt2_medium(),
        "large"  => Gpt2Config::gpt2_large(),
        "xl"     => Gpt2Config::gpt2_xl(),
        other    => anyhow::bail!("Unknown model size '{}'", other),
    }
    .with_dropout(0.0);

    let mut model: Gpt2Model<Backend> = cfg.init(&device);

    log::info!("Loading weights from '{}' in '{}' format...", model_path, args.format);
    if args.format == "safetensors" {
        let tensor_map = loader::load_safetensors(&model_path)?;
        model = loader::load_gpt2_weights(model, &tensor_map, &device)?;
    } else if args.format == "compact" {
        let record = CompactRecorder::new()
            .load(model_path.clone().into(), &device)
            .map_err(|e| anyhow::anyhow!("Failed to load compact checkpoint: {:?}", e))?;
        model = model.load_record(record);
    }

    let dataset = training::dataset::TextDataset::from_file(&args.data, args.seq_len)?;
    if dataset.len() == 0 {
        anyhow::bail!("Dataset is empty or too short.");
    }

    let batcher = training::dataset::TextBatcher;
    let dataloader = DataLoaderBuilder::<Backend, _, _>::new(batcher)
        .batch_size(args.batch_size)
        .num_workers(args.workers)
        .build(dataset);

    let mut loss_sum = 0.0f32;
    let mut batches_count = 0usize;

    for batch in dataloader.iter() {
        let logits = model.forward(batch.input_ids);
        let loss = training::loss::lm_loss(logits, batch.labels);
        let loss_val: f32 = loss
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0);

        loss_sum += loss_val;
        batches_count += 1;
    }

    let avg_loss = loss_sum / batches_count.max(1) as f32;
    let ppl = avg_loss.exp();

    println!("\n\x1b[1;36m=== Evaluation Results ===\x1b[0m");
    println!("Evaluation Dataset : \x1b[1;32m{}\x1b[0m", args.data);
    println!("Average Loss       : \x1b[1;33m{:.4}\x1b[0m", avg_loss);
    println!("Perplexity         : \x1b[1;32m{:.4}\x1b[0m", ppl);
    println!("\x1b[1;36m==========================\x1b[0m");
    Ok(())
}

// === Training Runner ===

fn run_train(args: &Args) -> Result<()> {
    let model_cfg = match args.size.as_str() {
        "small"  => Gpt2Config::gpt2_small(),
        "medium" => Gpt2Config::gpt2_medium(),
        "large"  => Gpt2Config::gpt2_large(),
        "xl"     => Gpt2Config::gpt2_xl(),
        other    => anyhow::bail!("Unknown size '{}'", other),
    }
    .with_dropout(args.dropout);

    let train_cfg = training::TrainingConfig::new(model_cfg)
        .with_num_epochs(args.epochs)
        .with_batch_size(args.batch_size)
        .with_seq_len(args.seq_len)
        .with_learning_rate(args.lr)
        .with_val_fraction(args.val_fraction)
        .with_seed(args.seed)
        .with_num_workers(args.workers)
        .with_train_data(args.data.clone());

    type MyBackend = Cuda<f32, i32>;
    type MyAutodiff = Autodiff<MyBackend>;
    let device = burn::backend::cuda::CudaDevice::default();

    training::train::<MyAutodiff>(&args.artifact_dir, train_cfg, device);
    Ok(())
}

// === Logger Helper ===

fn init_logger() {
    use std::io::Write;
    let mut builder = env_logger::Builder::from_default_env();
    if std::env::var("RUST_LOG").is_err() {
        builder.filter_level(log::LevelFilter::Info);
    }
    builder.write_style(env_logger::WriteStyle::Always);
    builder.format(|buf, record| {
        let level = record.level();
        let level_color = match level {
            log::Level::Error => "\x1b[1;31mERROR\x1b[0m",
            log::Level::Warn => "\x1b[1;33mWARN\x1b[0m",
            log::Level::Info => "\x1b[1;32mINFO\x1b[0m",
            log::Level::Debug => "\x1b[1;36mDEBUG\x1b[0m",
            log::Level::Trace => "\x1b[1;35mTRACE\x1b[0m",
        };
        writeln!(
            buf,
            "[🦀 GPT-2 {}] {}",
            level_color,
            record.args()
        )
    });
    builder.init();
}

// === CLI Main ===

fn main() -> Result<()> {
    init_logger();
    let args = Args::parse();

    if args.download {
        run_download(&args)?;
    } else if args.generate {
        run_generate(&args)?;
    } else if args.evaluate {
        run_eval(&args)?;
    } else if args.train {
        run_train(&args)?;
    } else {
        // Show help if no mode flag is supplied
        use clap::CommandFactory;
        Args::command().print_help()?;
        println!("\n\n\x1b[1;31mError: Please specify a mode flag: --download, --generate, --evaluate, or --train\x1b[0m");
    }
    Ok(())
}
