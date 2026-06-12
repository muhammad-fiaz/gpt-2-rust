//! GPT-2 model configuration.
//!
//! Provides [`Gpt2Config`] for all four official GPT-2 variants and any
//! custom configuration the user wants.

use burn::prelude::*;

/// Core hyper-parameters shared across every part of the GPT-2 model.
///
/// All four official sizes are available as associated constructors:
/// - [`Gpt2Config::gpt2_small`]  — 117 M params
/// - [`Gpt2Config::gpt2_medium`] — 345 M params
/// - [`Gpt2Config::gpt2_large`]  — 762 M params
/// - [`Gpt2Config::gpt2_xl`]     — 1.5 B params
#[derive(Config, Debug)]
pub struct Gpt2Config {
    /// Vocabulary size (GPT-2 BPE: 50 257).
    #[config(default = 50257)]
    pub vocab_size: usize,

    /// Maximum sequence length (context window).
    #[config(default = 1024)]
    pub max_seq_len: usize,

    /// Embedding / hidden dimension.
    #[config(default = 768)]
    pub n_embd: usize,

    /// Number of transformer blocks.
    #[config(default = 12)]
    pub n_layer: usize,

    /// Number of attention heads.  Must divide `n_embd` evenly.
    #[config(default = 12)]
    pub n_head: usize,

    /// Dropout probability applied to embeddings, attention weights, and the
    /// residual stream.  Set to 0.0 for inference.
    #[config(default = 0.1)]
    pub dropout: f64,

    /// Whether to use bias in `Linear` layers and `LayerNorm`.
    /// GPT-2 uses bias everywhere; set to `false` for GPT-3-style no-bias.
    #[config(default = true)]
    pub bias: bool,
}

impl Gpt2Config {
    // Official model presets

    /// GPT-2 Small — 117 M parameters.
    pub fn gpt2_small() -> Self {
        Self::new()
            .with_n_embd(768)
            .with_n_layer(12)
            .with_n_head(12)
    }

    /// GPT-2 Medium — 345 M parameters.
    pub fn gpt2_medium() -> Self {
        Self::new()
            .with_n_embd(1024)
            .with_n_layer(24)
            .with_n_head(16)
    }

    /// GPT-2 Large — 762 M parameters.
    pub fn gpt2_large() -> Self {
        Self::new()
            .with_n_embd(1280)
            .with_n_layer(36)
            .with_n_head(20)
    }

    /// GPT-2 XL — 1.5 B parameters.
    pub fn gpt2_xl() -> Self {
        Self::new()
            .with_n_embd(1600)
            .with_n_layer(48)
            .with_n_head(25)
    }

    // Derived helpers

    /// Head dimension: `n_embd / n_head`.
    #[inline]
    pub fn head_dim(&self) -> usize {
        assert_eq!(
            self.n_embd % self.n_head,
            0,
            "n_embd ({}) must be divisible by n_head ({})",
            self.n_embd,
            self.n_head
        );
        self.n_embd / self.n_head
    }

    /// MLP intermediate dimension: `4 * n_embd`.
    #[inline]
    pub fn mlp_dim(&self) -> usize {
        4 * self.n_embd
    }

    /// Initialise a [`Gpt2Model`] on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> crate::model::Gpt2Model<B> {
        crate::model::Gpt2Model::new(self, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_config() {
        let cfg = Gpt2Config::gpt2_small();
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.mlp_dim(), 3072);
    }

    #[test]
    fn test_medium_config() {
        let cfg = Gpt2Config::gpt2_medium();
        assert_eq!(cfg.n_embd, 1024);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn test_large_config() {
        let cfg = Gpt2Config::gpt2_large();
        assert_eq!(cfg.n_embd, 1280);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn test_xl_config() {
        let cfg = Gpt2Config::gpt2_xl();
        assert_eq!(cfg.n_embd, 1600);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    #[should_panic]
    fn test_bad_head_divisor() {
        Gpt2Config::new()
            .with_n_embd(768)
            .with_n_head(7) // 768 % 7 != 0
            .head_dim();
    }
}
