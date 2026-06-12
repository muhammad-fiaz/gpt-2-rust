//! Text dataset and batcher for language-model pre-training.
//!
//! Loads a plain-text file, tokenises it with the GPT-2 BPE tokenizer, and
//! serves sliding-window `(input, label)` pairs to the training loop.

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

use crate::tokenizer::Gpt2Tokenizer;

/// A single training item: a `seq_len`-token context window.
#[derive(Debug, Clone)]
pub struct TextItem {
    /// Token IDs for the input window.
    pub input_ids: Vec<u32>,
    /// Token IDs for the target window (shifted right by 1).
    pub labels: Vec<u32>,
}

/// In-memory dataset backed by a tokenised text file.
///
/// Constructs sliding windows of length `seq_len + 1` (the `+1` is needed to
/// produce the next-token label) with a stride of 1.
pub struct TextDataset {
    items: Vec<TextItem>,
}

impl TextDataset {
    /// Build a dataset from a raw text string.
    pub fn from_text(text: &str, seq_len: usize) -> Self {
        let tokenizer = Gpt2Tokenizer::default();
        let all_tokens = tokenizer.encode(text);

        let items = all_tokens
            .windows(seq_len + 1)
            .map(|window| TextItem {
                input_ids: window[..seq_len].to_vec(),
                labels: window[1..=seq_len].to_vec(),
            })
            .collect();

        Self { items }
    }

    /// Build a dataset from a file path.
    pub fn from_file(path: &str, seq_len: usize) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", path, e))?;
        Ok(Self::from_text(&text, seq_len))
    }
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// A batched set of training examples.
#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    /// Input token IDs `[batch_size, seq_len]`.
    pub input_ids: Tensor<B, 2, Int>,
    /// Target token IDs `[batch_size, seq_len]`.
    pub labels: Tensor<B, 2, Int>,
}

/// Collates `TextItem` values into a `TextBatch`.
#[derive(Clone, Default)]
pub struct TextBatcher;

impl<B: Backend> Batcher<B, TextItem, TextBatch<B>> for TextBatcher {
    fn batch(&self, items: Vec<TextItem>, device: &B::Device) -> TextBatch<B> {
        let seq_len = items[0].input_ids.len();

        let input_tensors: Vec<Tensor<B, 2, Int>> = items
            .iter()
            .map(|item| {
                let ids: Vec<i64> = item.input_ids.iter().map(|&t| t as i64).collect();
                Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [seq_len]), device)
                    .unsqueeze::<2>() // [1, seq_len]
            })
            .collect();

        let label_tensors: Vec<Tensor<B, 2, Int>> = items
            .iter()
            .map(|item| {
                let ids: Vec<i64> = item.labels.iter().map(|&t| t as i64).collect();
                Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [seq_len]), device)
                    .unsqueeze::<2>()
            })
            .collect();

        let input_ids = Tensor::cat(input_tensors, 0); // [B, T]
        let labels = Tensor::cat(label_tensors, 0); // [B, T]

        TextBatch { input_ids, labels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_length() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let ds = TextDataset::from_text(&text, 8);
        let tokens = Gpt2Tokenizer::default().encode(&text);
        // Expected: len(tokens) - seq_len windows
        assert_eq!(ds.len(), tokens.len().saturating_sub(8));
    }

    #[test]
    fn test_item_shape() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(5);
        let ds = TextDataset::from_text(&text, 16);
        if let Some(item) = ds.get(0) {
            assert_eq!(item.input_ids.len(), 16);
            assert_eq!(item.labels.len(), 16);
            // labels[i] == input_ids[i+1] (next-token shift)
            assert_eq!(item.labels[0], item.input_ids[1]);
        }
    }
}
