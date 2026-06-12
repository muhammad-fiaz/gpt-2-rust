//! Language-model loss — cross-entropy over the next-token prediction.

use burn::{
    nn::loss::CrossEntropyLossConfig,
    prelude::*,
};

/// Compute the cross-entropy language-model loss.
///
/// Logits and labels must be aligned: `labels[t]` is the token that should
/// follow `input_ids[t]` (i.e. they are already pre-shifted by the dataset).
///
/// # Arguments
/// - `logits` — `[batch, seq_len, vocab_size]`
/// - `labels` — `[batch, seq_len]` (Int)
///
/// # Returns
/// Scalar mean loss over all (batch × time) positions.
pub fn lm_loss<B: Backend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch, seq_len, vocab_size] = logits.dims();

    // Flatten for cross-entropy: [B*T, V] and [B*T]
    let logits_flat = logits.reshape([batch * seq_len, vocab_size]);
    let labels_flat = labels.reshape([batch * seq_len]);

    CrossEntropyLossConfig::new()
        .init(&logits_flat.device())
        .forward(logits_flat, labels_flat)
}

/// Convenience: compute perplexity from a loss scalar.
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_loss_shape() {
        let device = Default::default();
        // batch=2, seq=4, vocab=10
        let logits = Tensor::<B, 3>::ones([2, 4, 10], &device);
        let labels = Tensor::<B, 2, Int>::zeros([2, 4], &device);
        let loss = lm_loss(logits, labels);
        assert_eq!(loss.dims(), [1]);
    }

    #[test]
    fn test_loss_is_positive() {
        let device = Default::default();
        let logits = Tensor::<B, 3>::random(
            [2, 4, 10],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let labels = Tensor::<B, 2, Int>::zeros([2, 4], &device);
        let loss_val: f32 = lm_loss(logits, labels)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss_val > 0.0);
    }
}
