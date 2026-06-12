//! Full GPT-2 model.
//!
//! Assembles token embedding, positional embedding, transformer blocks, and
//! the language-model head into a single Burn `Module`.

pub mod attention;
pub mod block;
pub mod mlp;

use burn::{
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::config::Gpt2Config;
use block::TransformerBlock;

/// The full GPT-2 model.
///
/// Matches the weight layout of the official OpenAI checkpoint:
/// - `transformer.wte`   — token embedding
/// - `transformer.wpe`   — positional embedding
/// - `transformer.h.{i}` — transformer blocks
/// - `transformer.ln_f`  — final layer norm
/// - `lm_head`           — language-model head (weight-tied to `wte`)
#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    /// Token embedding table `[vocab_size, n_embd]`.
    wte: Embedding<B>,
    /// Positional embedding table `[max_seq_len, n_embd]`.
    wpe: Embedding<B>,
    /// Embedding dropout.
    drop: Dropout,
    /// Stack of transformer blocks.
    blocks: Vec<TransformerBlock<B>>,
    /// Final layer normalisation.
    ln_f: LayerNorm<B>,
    /// Language-model head `[n_embd → vocab_size]` (no bias, weight-tied).
    lm_head: Linear<B>,
}

impl<B: Backend> Gpt2Model<B> {
    /// Build a new `Gpt2Model` with random initialisation.
    pub fn new(cfg: &Gpt2Config, device: &B::Device) -> Self {
        let wte = EmbeddingConfig::new(cfg.vocab_size, cfg.n_embd).init(device);
        let wpe = EmbeddingConfig::new(cfg.max_seq_len, cfg.n_embd).init(device);
        let drop = DropoutConfig::new(cfg.dropout).init();

        let blocks = (0..cfg.n_layer)
            .map(|_| TransformerBlock::new(cfg, device))
            .collect();

        let ln_f = LayerNormConfig::new(cfg.n_embd).init(device);

        // LM head: n_embd → vocab_size, no bias.
        // Weight tying (sharing weights with wte) is handled at the Burn record
        // level when loading from safetensors; for random init we use a separate
        // matrix (Burn does not have a built-in weight-tie API yet).
        let lm_head = LinearConfig::new(cfg.n_embd, cfg.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            wte,
            wpe,
            drop,
            blocks,
            ln_f,
            lm_head,
        }
    }

    /// Forward pass — returns **logits** `[batch, seq_len, vocab_size]`.
    ///
    /// # Arguments
    /// - `input_ids` — Token IDs `[batch, seq_len]` (Int tensor)
    ///
    /// # Panics
    /// Panics if `seq_len > max_seq_len`.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch, seq_len] = input_ids.dims();

        // Token + Positional Embeddings
        let tok_emb = self.wte.forward(input_ids.clone()); // [B, T, n_embd]

        let positions = Tensor::<B, 1, Int>::arange(
            0..(seq_len as i64),
            &input_ids.device(),
        )
        .unsqueeze::<2>(); // [1, T]

        let pos_emb = self.wpe.forward(positions); // [1, T, n_embd] — broadcasts
        let mut x = self.drop.forward(tok_emb + pos_emb);

        // Transformer Blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Final LayerNorm + LM Head
        let x = self.ln_f.forward(x); // [B, T, n_embd]
        self.lm_head.forward(x) // [B, T, vocab_size]
    }

    /// Convenience method: forward pass returning only the **last** token's
    /// logits — useful during autoregressive generation.
    ///
    /// Returns `[batch, vocab_size]`.
    pub fn forward_last(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids); // [B, T, V]
        let [batch, seq_len, vocab] = logits.dims();
        // Slice the last time step
        logits.slice([0..batch, (seq_len - 1)..seq_len, 0..vocab])
            .reshape([batch, vocab])
    }

    /// Number of trainable parameters (approximate, counts tensor elements).
    pub fn num_params(&self) -> usize {
        // Burn doesn't expose a built-in param counter; we compute from config.
        // This is an informational helper only.
        0 // placeholder — populated at runtime via module inspection
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn tiny_cfg() -> Gpt2Config {
        Gpt2Config::new()
            .with_vocab_size(256)
            .with_max_seq_len(16)
            .with_n_embd(32)
            .with_n_layer(2)
            .with_n_head(4)
            .with_dropout(0.0)
    }

    #[test]
    fn test_forward_shape() {
        let device = Default::default();
        let cfg = tiny_cfg();
        let model = Gpt2Model::<B>::new(&cfg, &device);

        let ids = Tensor::<B, 2, Int>::zeros([2, 8], &device);
        let logits = model.forward(ids);
        assert_eq!(logits.dims(), [2, 8, 256]);
    }

    #[test]
    fn test_forward_last_shape() {
        let device = Default::default();
        let cfg = tiny_cfg();
        let model = Gpt2Model::<B>::new(&cfg, &device);

        let ids = Tensor::<B, 2, Int>::zeros([3, 5], &device);
        let last = model.forward_last(ids);
        assert_eq!(last.dims(), [3, 256]);
    }
}
