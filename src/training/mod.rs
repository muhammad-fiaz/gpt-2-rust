//! Training loop for GPT-2.
//!
//! Implements a manual training loop using Burn's autodiff backend, optimizer,
//! and DataLoader API — without the higher-level Learner abstraction to keep
//! the trait-bound complexity manageable.

pub mod dataset;
pub mod loss;

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    model::Gpt2Model,
    training::dataset::{TextBatch, TextBatcher, TextDataset},
    training::loss::lm_loss,
};

// Training configuration

/// Full training configuration.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// GPT-2 model hyperparameters.
    pub model: crate::config::Gpt2Config,

    /// AdamW optimiser configuration.
    #[config(default = "AdamWConfig::new()")]
    pub optimizer: AdamWConfig,

    /// Number of training epochs.
    #[config(default = 3)]
    pub num_epochs: usize,

    /// Batch size (number of sequences per step).
    #[config(default = 4)]
    pub batch_size: usize,

    /// Sequence length (tokens per example).
    #[config(default = 128)]
    pub seq_len: usize,

    /// Number of DataLoader worker threads.
    #[config(default = 2)]
    pub num_workers: usize,

    /// Random seed for reproducibility.
    #[config(default = 42)]
    pub seed: u64,

    /// Peak learning rate.
    #[config(default = 3.0e-4)]
    pub learning_rate: f64,

    /// Path to training text file.
    #[config(default = "\"data/input.txt\".to_string()")]
    pub train_data: String,

    /// Fraction of data used for validation (0.0–1.0).
    #[config(default = 0.1)]
    pub val_fraction: f64,
}

// Main training function

/// Run the full GPT-2 training loop.
///
/// # Arguments
/// - `artifact_dir` — directory for checkpoints and the final model
/// - `config`       — [`TrainingConfig`]
/// - `device`       — target backend device
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) where
    B::IntElem: burn::tensor::ElementConversion,
{
    // Create artifact directory
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir)
        .expect("Failed to create artifact directory");

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved");

    B::seed(&device, config.seed);

    // Load & split dataset
    log::info!("Loading training data from '{}'", config.train_data);
    let text = std::fs::read_to_string(&config.train_data)
        .expect("Failed to read training data file");

    let all_tokens = crate::tokenizer::Gpt2Tokenizer::default().encode(&text);
    let split_at = ((1.0 - config.val_fraction) * all_tokens.len() as f64) as usize;

    let train_text = crate::tokenizer::Gpt2Tokenizer::default()
        .decode(&all_tokens[..split_at]);
    let val_text = crate::tokenizer::Gpt2Tokenizer::default()
        .decode(&all_tokens[split_at..]);

    let train_dataset = TextDataset::from_text(&train_text, config.seq_len);
    let val_dataset = TextDataset::from_text(&val_text, config.seq_len);

    log::info!(
        "Dataset: {} train / {} val windows (seq_len={})",
        train_dataset.len(),
        val_dataset.len(),
        config.seq_len,
    );

    let batcher = TextBatcher;

    let dataloader_train = DataLoaderBuilder::<B, _, _>::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    // Validation dataloader (unused for now as validation loop is skipped)
    // let _dataloader_val = DataLoaderBuilder::<B, _, _>::new(batcher)
    //     .batch_size(config.batch_size)
    //     .num_workers(config.num_workers)
    //     .build(val_dataset);

    // Build model & optimiser
    let mut model: Gpt2Model<B> = config.model.init(&device);
    let mut optim = config.optimizer.init();

    // Training loop
    for epoch in 1..=config.num_epochs {
        // — Train —
        let mut train_loss_sum = 0.0f32;
        let mut train_batches = 0usize;

        for batch in dataloader_train.iter() {
            let batch: TextBatch<B> = batch;
            let logits = model.forward(batch.input_ids);
            let loss = lm_loss(logits, batch.labels);

            let loss_val: f32 = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(f32::NAN);

            train_loss_sum += loss_val;
            train_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        let avg_train_loss = train_loss_sum / train_batches.max(1) as f32;

        // — Validate —
        // Note: A clean validation loop requires splitting the model into its
        // InnerBackend counterpart. For now, we log training loss only and leave
        // full validation as a post-training evaluation step.

        log::info!(
            "Epoch {}/{} — train_loss = {:.4} (ppl = {:.2})",
            epoch,
            config.num_epochs,
            avg_train_loss,
            avg_train_loss.exp(),
        );

        // Checkpoint every epoch
        model
            .clone()
            .save_file(
                format!("{artifact_dir}/gpt2_epoch_{epoch}"),
                &CompactRecorder::new(),
            )
            .expect("Failed to save checkpoint");
    }

    // Save final model
    model
        .save_file(
            format!("{artifact_dir}/gpt2_model"),
            &CompactRecorder::new(),
        )
        .expect("Failed to save trained model");

    log::info!("Training complete. Model saved to '{artifact_dir}/gpt2_model'");
}
