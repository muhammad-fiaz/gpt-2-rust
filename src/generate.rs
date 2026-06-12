//! Text generation with temperature, top-k, and top-p (nucleus) sampling.
//!
//! The [`generate`] function runs the autoregressive loop and supports:
//! - **Greedy** decoding (`temperature = 0`)
//! - **Temperature** scaling
//! - **Top-K** filtering
//! - **Top-P** (nucleus) filtering
//! - Early stopping on the EOT token

use burn::prelude::*;

use serde::{Deserialize, Serialize};

use crate::model::Gpt2Model;
use crate::tokenizer::{EOT_TOKEN, Gpt2Tokenizer};

/// Configuration for autoregressive text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of **new** tokens to generate (not counting the prompt).
    pub max_new_tokens: usize,

    /// Temperature for logit scaling.
    /// - `1.0` → unchanged distribution
    /// - `< 1.0` → sharper (more deterministic)
    /// - `> 1.0` → flatter (more random)
    /// - `0.0` → greedy argmax
    pub temperature: f64,

    /// If set, keep only the `k` highest-probability tokens before sampling.
    pub top_k: Option<usize>,

    /// If set, keep the smallest set of tokens whose cumulative probability
    /// is at least `p` (nucleus sampling).
    pub top_p: Option<f64>,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Stop early when the EOT token is generated.
    pub stop_on_eot: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 200,
            temperature: 0.8,
            top_k: Some(50),
            top_p: None,
            seed: 42,
            stop_on_eot: true,
        }
    }
}

/// Generate text autoregressively.
///
/// # Arguments
/// - `model`     — Initialised [`Gpt2Model`]
/// - `tokenizer` — GPT-2 BPE tokenizer
/// - `prompt`    — Starting text
/// - `config`    — Sampling configuration
/// - `device`    — Target device
///
/// # Returns
/// The full generated string (prompt + new tokens).
pub fn generate<B: Backend>(
    model: &Gpt2Model<B>,
    tokenizer: &Gpt2Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    device: &B::Device,
) -> String {
    use rand::{rngs::StdRng, SeedableRng};


    let mut rng = StdRng::seed_from_u64(config.seed);

    // Encode prompt
    let mut token_ids: Vec<u32> = tokenizer.encode(prompt);

    if token_ids.is_empty() {
        token_ids.push(tokenizer.eot_token());
    }

    // Autoregressive loop
    for _ in 0..config.max_new_tokens {
        // Build input tensor [1, T]
        let ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();
        let input = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids_i64, [token_ids.len()]),
            device,
        )
        .unsqueeze::<2>(); // [1, T]

        // Forward pass — get logits for last position [1, vocab_size]
        let logits = model.forward_last(input); // [1, V]

        // Extract as f32 vec
        let mut logits_vec: Vec<f32> = logits
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to convert logits to f32");

        // Temperature scaling
        if config.temperature > 0.0 && config.temperature != 1.0 {
            let inv_temp = (1.0 / config.temperature) as f32;
            for l in logits_vec.iter_mut() {
                *l *= inv_temp;
            }
        }

        // Top-K filtering
        if let Some(k) = config.top_k {
            if k < logits_vec.len() {
                top_k_filter(&mut logits_vec, k);
            }
        }

        // Top-P (nucleus) filtering
        if let Some(p) = config.top_p {
            top_p_filter(&mut logits_vec, p as f32);
        }

        // Sampling
        let next_token = if config.temperature == 0.0 {
            // Greedy argmax
            logits_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0) as u32
        } else {
            sample_from_logits(&logits_vec, &mut rng)
        };

        // Stream the token's text representation
        let token_text = tokenizer.decode(&[next_token]);
        print!("{}", token_text);
        use std::io::Write;
        let _ = std::io::stdout().flush();

        token_ids.push(next_token);

        if config.stop_on_eot && next_token == EOT_TOKEN {
            break;
        }
    }

    tokenizer.decode(&token_ids)
}

// Sampling helpers

/// Set logits below the k-th largest to −∞ in-place.
fn top_k_filter(logits: &mut Vec<f32>, k: usize) {
    let mut sorted = logits.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let threshold = sorted[k - 1];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Apply nucleus (top-p) filtering in-place.
fn top_p_filter(logits: &mut Vec<f32>, p: f32) {
    // Build (index, logit) sorted by logit descending
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Softmax over sorted logits
    let max_l = indexed[0].1;
    let mut sum_exp = 0.0f32;
    let exps: Vec<f32> = indexed
        .iter()
        .map(|(_, l)| {
            let e = (l - max_l).exp();
            sum_exp += e;
            e
        })
        .collect();

    // Accumulate until cumulative probability >= p
    let mut cum_prob = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (rank, &(_, _)) in indexed.iter().enumerate() {
        cum_prob += exps[rank] / sum_exp;
        if cum_prob >= p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    // Mask out everything below the nucleus
    let nucleus: std::collections::HashSet<usize> = indexed[..cutoff_idx]
        .iter()
        .map(|(i, _)| *i)
        .collect();

    for (i, l) in logits.iter_mut().enumerate() {
        if !nucleus.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Sample a token index from logits using categorical sampling.
fn sample_from_logits<R: rand::Rng>(logits: &[f32], rng: &mut R) -> u32 {
    // Stable softmax
    let max_l = logits
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut exps: Vec<f32> = logits
        .iter()
        .map(|&l| if l.is_finite() { (l - max_l).exp() } else { 0.0 })
        .collect();

    let sum: f32 = exps.iter().sum();
    for e in exps.iter_mut() {
        *e /= sum;
    }

    // Inverse CDF sampling
    let u: f32 = rng.r#gen::<f32>();
    let mut cum = 0.0f32;
    for (i, &prob) in exps.iter().enumerate() {
        cum += prob;
        if u <= cum {
            return i as u32;
        }
    }
    (exps.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_filter() {
        let mut logits = vec![1.0, 3.0, 2.0, 0.5, 4.0];
        top_k_filter(&mut logits, 2);
        // Top-2 are 4.0 and 3.0; rest → -inf
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert_eq!(logits[1], 3.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
        assert_eq!(logits[4], 4.0);
    }

    #[test]
    fn test_top_p_filter_keeps_top() {
        let mut logits = vec![100.0, -1.0, -1.0, -1.0];
        top_p_filter(&mut logits, 0.9);
        // First logit dominates; rest should be masked
        assert_eq!(logits[0], 100.0);
        for &l in &logits[1..] {
            assert!(l.is_infinite() && l < 0.0);
        }
    }

    #[test]
    fn test_sample_from_logits_deterministic() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0);
        let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 100.0, f32::NEG_INFINITY];
        // Only index 2 has non-zero probability
        assert_eq!(sample_from_logits(&logits, &mut rng), 2);
    }
}
