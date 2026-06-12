//! Causal self-attention module for GPT-2.
//!
//! This is a **custom** implementation — not a thin wrapper around
//! `burn::nn::MultiHeadAttention` — so we can match GPT-2's exact weight
//! layout (single `c_attn` projection for Q/K/V) and causal masking semantics.

use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::softmax,
};

use crate::config::Gpt2Config;

/// Multi-head causal (decoder-only) self-attention.
///
/// Weight layout matches OpenAI's GPT-2 checkpoint:
/// - `c_attn` — combined Q/K/V projection `[n_embd → 3*n_embd]`
/// - `c_proj` — output projection `[n_embd → n_embd]`
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    /// Combined Q, K, V projection (split at runtime).
    c_attn: Linear<B>,
    /// Output projection.
    c_proj: Linear<B>,
    /// Attention-weight dropout.
    attn_drop: Dropout,
    /// Residual dropout.
    resid_drop: Dropout,
    /// Number of attention heads.
    n_head: usize,
    /// Head dimension: n_embd / n_head.
    head_dim: usize,
    /// Maximum sequence length (for pre-computing causal mask shape).
    max_seq_len: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    /// Construct a new `CausalSelfAttention` layer.
    pub fn new(cfg: &Gpt2Config, device: &B::Device) -> Self {
        let n_embd = cfg.n_embd;
        let head_dim = cfg.head_dim();

        // Single matrix that projects input to Q, K, V concatenated.
        let c_attn = LinearConfig::new(n_embd, 3 * n_embd)
            .with_bias(cfg.bias)
            .init(device);

        let c_proj = LinearConfig::new(n_embd, n_embd)
            .with_bias(cfg.bias)
            .init(device);

        let attn_drop = DropoutConfig::new(cfg.dropout).init();
        let resid_drop = DropoutConfig::new(cfg.dropout).init();

        Self {
            c_attn,
            c_proj,
            attn_drop,
            resid_drop,
            n_head: cfg.n_head,
            head_dim,
            max_seq_len: cfg.max_seq_len,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x` — Input tensor `[batch, seq_len, n_embd]`
    ///
    /// # Returns
    /// Output tensor `[batch, seq_len, n_embd]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _n_embd] = x.dims();

        // 1. Combined QKV projection
        // x: [B, T, n_embd]  →  qkv: [B, T, 3*n_embd]
        let qkv = self.c_attn.forward(x);

        // Split along last dimension into Q, K, V each [B, T, n_embd]
        let n_embd = self.n_head * self.head_dim;
        let q = qkv.clone().slice([0..batch, 0..seq_len, 0..n_embd]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq_len, n_embd..2 * n_embd]);
        let v = qkv.slice([0..batch, 0..seq_len, 2 * n_embd..3 * n_embd]);

        // 2. Reshape to multi-head form
        // [B, T, n_embd] → [B, T, n_head, head_dim] → [B, n_head, T, head_dim]
        let q = q
            .reshape([batch, seq_len, self.n_head, self.head_dim])
            .swap_dims(1, 2); // [B, n_head, T, head_dim]
        let k = k
            .reshape([batch, seq_len, self.n_head, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.n_head, self.head_dim])
            .swap_dims(1, 2);

        // 3. Scaled dot-product attention + causal mask
        // scores: [B, n_head, T, T]
        let scale = (self.head_dim as f64).sqrt();
        // k^T: [B, n_head, head_dim, T]
        let k_t = k.transpose();
        let scores = q.matmul(k_t) / scale;

        // Causal mask: upper triangle is -inf so future tokens are invisible.
        // mask[i,j] = -inf if j > i
        let mask = self.causal_mask(seq_len, &scores.device());
        let scores = scores + mask;

        // Softmax over the last (key) dimension
        let attn_weights = softmax(scores, 3); // [B, n_head, T, T]
        let attn_weights = self.attn_drop.forward(attn_weights);

        // Weighted sum of values
        // [B, n_head, T, T] × [B, n_head, T, head_dim] → [B, n_head, T, head_dim]
        let context = attn_weights.matmul(v);

        // 4. Re-assemble heads
        // [B, n_head, T, head_dim] → [B, T, n_head, head_dim] → [B, T, n_embd]
        let context = context
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_head * self.head_dim]);

        // 5. Output projection
        let out = self.c_proj.forward(context);
        self.resid_drop.forward(out)
    }

    /// Build a lower-triangular causal mask of shape `[1, 1, T, T]`.
    ///
    /// Positions where `j > i` are set to `−∞` (–1e9 in f32 to avoid
    /// NaN in softmax with fp16).
    fn causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        // Create index tensors [T]
        let rows = Tensor::<B, 1, Int>::arange(0..(seq_len as i64), device)
            .reshape([seq_len, 1]);
        let cols = Tensor::<B, 1, Int>::arange(0..(seq_len as i64), device)
            .reshape([1, seq_len]);

        // Boolean mask: True where col > row (future positions)
        let future_mask = cols.greater(rows); // [T, T]

        // Convert: True → -1e9, False → 0.0
        let neg_inf = Tensor::<B, 2>::zeros([seq_len, seq_len], device);
        let mask_2d = neg_inf.mask_fill(future_mask, -1e9_f32);

        // Expand to [1, 1, T, T] for broadcasting over batch and heads
        mask_2d.unsqueeze::<4>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_attention_output_shape() {
        let device = Default::default();
        let cfg = crate::config::Gpt2Config::new()
            .with_n_embd(64)
            .with_n_head(4)
            .with_n_layer(1)
            .with_max_seq_len(16)
            .with_dropout(0.0);

        let attn = CausalSelfAttention::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::zeros([2, 8, 64], &device);
        let out = attn.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_causal_mask_shape() {
        let device = Default::default();
        let cfg = crate::config::Gpt2Config::new()
            .with_n_embd(64)
            .with_n_head(4)
            .with_n_layer(1)
            .with_max_seq_len(16)
            .with_dropout(0.0);

        let attn = CausalSelfAttention::<B>::new(&cfg, &device);
        let mask = attn.causal_mask(8, &device);
        assert_eq!(mask.dims(), [1, 1, 8, 8]);
    }
}
