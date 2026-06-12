//! MLP (Feed-Forward Network) block for GPT-2.
//!
//! Implements the two-layer MLP used inside every transformer block:
//! `n_embd → 4*n_embd → n_embd` with approximate GELU activation.

use burn::{
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    prelude::*,
};

use crate::config::Gpt2Config;

/// Position-wise feed-forward network.
///
/// Architecture:
/// ```text
/// x → fc1 (n_embd → 4*n_embd) → GELU → fc2 (4*n_embd → n_embd) → dropout
/// ```
///
/// Weight names mirror the OpenAI checkpoint: `c_fc` and `c_proj`.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// First linear layer (expansion).
    c_fc: Linear<B>,
    /// Activation function (approximate GELU, as used in GPT-2).
    act: Gelu,
    /// Second linear layer (projection back to n_embd).
    c_proj: Linear<B>,
    /// Residual dropout.
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    /// Create a new MLP block.
    pub fn new(cfg: &Gpt2Config, device: &B::Device) -> Self {
        let n_embd = cfg.n_embd;
        let mlp_dim = cfg.mlp_dim();

        let c_fc = LinearConfig::new(n_embd, mlp_dim)
            .with_bias(cfg.bias)
            .init(device);

        let c_proj = LinearConfig::new(mlp_dim, n_embd)
            .with_bias(cfg.bias)
            .init(device);

        let dropout = DropoutConfig::new(cfg.dropout).init();

        Self {
            c_fc,
            act: Gelu::new(),
            c_proj,
            dropout,
        }
    }

    /// Forward pass: `[B, T, n_embd] → [B, T, n_embd]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = self.act.forward(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_mlp_output_shape() {
        let device = Default::default();
        let cfg = crate::config::Gpt2Config::new()
            .with_n_embd(64)
            .with_n_head(4)
            .with_n_layer(1)
            .with_dropout(0.0);

        let mlp = Mlp::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::zeros([2, 8, 64], &device);
        let out = mlp.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }
}
