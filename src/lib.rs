//! `gpt-2-rust` — GPT-2 implemented from scratch in Rust using Burn 0.21.
//!
//! ## Citations & Project Information
//! - **Repository:** [https://github.com/muhammad-fiaz/gpt-2-rust](https://github.com/muhammad-fiaz/gpt-2-rust)
//! - **Author:** Muhammad Fiaz (contact@muhammadfiaz.com)
//! - **Year:** 2026
//! - **License:** MIT License
//!
//! ## Architecture
//! - **Backend:** CUDA (native Rust GPU support via CubeCL and cudarc)
//! - **Framework:** [Burn](https://burn.dev) 0.21
//! - **Tokenizer:** Pure-Rust BPE (`tiktoken-rs` / GPT-2 `r50k_base`)
//! - **Weights:** Native safetensors weight loading
//!
//! ## Quick start
//! ```bash
//! # Generate text (requires downloaded weights)
//! cargo run --release -- --generate \
//!   --model weights/small/model.safetensors \
//!   --prompt "The future of AI is"
//!
//! # Fine-tune on input data
//! cargo run --release -- --train \
//!   --data data/input.txt \
//!   --artifact-dir artifacts/
//! ```

pub mod config;
pub mod generate;
pub mod loader;
pub mod model;
pub mod tokenizer;
pub mod training;

// Re-export the most commonly used types at the crate root.
pub use config::Gpt2Config;
pub use generate::{generate, GenerationConfig};
pub use model::Gpt2Model;
pub use tokenizer::Gpt2Tokenizer;
