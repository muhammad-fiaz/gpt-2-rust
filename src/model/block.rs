//! GPT-2 transformer block.
//!
//! Each block follows the **pre-norm** variant used in GPT-2:
//! ```text
//! x = x + Attention(LayerNorm(x))
//! x = x + MLP(LayerNorm(x))
//! ```

use burn::{
    nn::{LayerNorm, LayerNormConfig},
    prelude::*,
};

use crate::{
    config::Gpt2Config,
    model::{attention::CausalSelfAttention, mlp::Mlp},
};

/// A single GPT-2 transformer block (pre-norm, residual attention + MLP).
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Pre-attention layer norm.
    ln_1: LayerNorm<B>,
    /// Causal self-attention.
    attn: CausalSelfAttention<B>,
    /// Pre-MLP layer norm.
    ln_2: LayerNorm<B>,
    /// Feed-forward network.
    mlp: Mlp<B>,
}

impl<B: Backend> TransformerBlock<B> {
    /// Construct a transformer block.
    pub fn new(cfg: &Gpt2Config, device: &B::Device) -> Self {
        let ln_1 = LayerNormConfig::new(cfg.n_embd).init(device);
        let attn = CausalSelfAttention::new(cfg, device);
        let ln_2 = LayerNormConfig::new(cfg.n_embd).init(device);
        let mlp = Mlp::new(cfg, device);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x` — `[batch, seq_len, n_embd]`
    ///
    /// # Returns
    /// `[batch, seq_len, n_embd]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention with pre-norm and residual
        let residual = x.clone();
        let x = self.ln_1.forward(x);
        let x = self.attn.forward(x);
        let x = residual + x;

        // MLP with pre-norm and residual
        let residual = x.clone();
        let x = self.ln_2.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_block_output_shape() {
        let device = Default::default();
        let cfg = crate::config::Gpt2Config::new()
            .with_n_embd(64)
            .with_n_head(4)
            .with_n_layer(2)
            .with_max_seq_len(16)
            .with_dropout(0.0);

        let block = TransformerBlock::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::zeros([2, 8, 64], &device);
        let out = block.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }
}
