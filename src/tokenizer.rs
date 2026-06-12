//! BPE tokenizer wrapper for GPT-2.
//!
//! Wraps [`tiktoken_rs`] to provide a simple encode/decode API compatible
//! with GPT-2's 50 257-token vocabulary (the `r50k_base` / `gpt2` encoding).

use anyhow::{Context, Result};
use tiktoken_rs::CoreBPE;

/// GPT-2 vocabulary size.
pub const VOCAB_SIZE: usize = 50257;

/// End-of-text token id in the GPT-2 vocabulary.
pub const EOT_TOKEN: u32 = 50256;

/// Thin, `Send + Sync` wrapper around the tiktoken GPT-2 BPE tokenizer.
pub struct Gpt2Tokenizer {
    bpe: CoreBPE,
}

impl Gpt2Tokenizer {
    /// Construct a tokenizer with the GPT-2 vocabulary (`r50k_base`).
    ///
    /// Falls back to `cl100k_base` if the GPT-2 encoding is unavailable;
    /// use `from_tiktoken` for a strictly correct GPT-2 tokenizer.
    pub fn new() -> Result<Self> {
        // tiktoken-rs ships several encodings; r50k_base == GPT-2
        let bpe = tiktoken_rs::r50k_base()
            .context("Failed to load GPT-2 (r50k_base) BPE tokenizer")?;
        Ok(Self { bpe })
    }

    /// Encode a text string into a list of token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe
            .encode_with_special_tokens(text)
            .into_iter()
            .map(|t| t as u32)
            .collect()
    }

    /// Decode a list of token IDs back into a UTF-8 string.
    ///
    /// Invalid byte sequences are replaced with the Unicode replacement
    /// character (`\u{FFFD}`).
    pub fn decode(&self, tokens: &[u32]) -> String {
        let ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        self.bpe
            .decode(ids)
            .unwrap_or_else(|_| String::from("\u{FFFD}"))
    }

    /// The vocabulary size (50 257 for GPT-2).
    #[inline]
    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    /// The end-of-text special token ID.
    #[inline]
    pub fn eot_token(&self) -> u32 {
        EOT_TOKEN
    }
}

impl Default for Gpt2Tokenizer {
    fn default() -> Self {
        Self::new().expect("Failed to initialise GPT-2 tokenizer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip() {
        let tok = Gpt2Tokenizer::new().unwrap();
        let text = "Hello, world! This is a GPT-2 test.";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size() {
        let tok = Gpt2Tokenizer::new().unwrap();
        assert_eq!(tok.vocab_size(), 50257);
    }

    #[test]
    fn test_eot() {
        let tok = Gpt2Tokenizer::new().unwrap();
        assert_eq!(tok.eot_token(), 50256);
    }

    #[test]
    fn test_nonempty_encoding() {
        let tok = Gpt2Tokenizer::new().unwrap();
        let ids = tok.encode("Burn is amazing.");
        assert!(!ids.is_empty());
        // All ids must be in range
        for id in &ids {
            assert!(*id < VOCAB_SIZE as u32);
        }
    }
}
