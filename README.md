<h1 align="center">GPT-2 Rust 🦀🔥</h1>

<p align="center">
  <strong>A native Rust implementation of the GPT-2 transformer architecture built from scratch using the Burn deep-learning framework.</strong>
</p>

<p align="center">
  <a href="https://github.com/muhammad-fiaz/gpt-2-rust/actions"><img src="https://img.shields.io/github/actions/workflow/status/muhammad-fiaz/gpt-2-rust/rust.yml?branch=main" alt="Build Status" /></a>
  <a href="https://github.com/muhammad-fiaz/gpt-2-rust/issues"><img src="https://img.shields.io/github/issues/muhammad-fiaz/gpt-2-rust" alt="GitHub Issues" /></a>
  <a href="https://github.com/muhammad-fiaz/gpt-2-rust/pulls"><img src="https://img.shields.io/github/issues-pr/muhammad-fiaz/gpt-2-rust" alt="GitHub Pull Requests" /></a>
  <a href="https://github.com/muhammad-fiaz/gpt-2-rust/graphs/commit-activity"><img src="https://img.shields.io/github/last-commit/muhammad-fiaz/gpt-2-rust" alt="GitHub Last Commit" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/muhammad-fiaz/gpt-2-rust" alt="License: MIT" /></a>
</p>

---

## Overview

This project is a native Rust implementation of the Transformer model from the seminal paper [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762) / [(Hugging Face)](https://huggingface.co/papers/1706.03762) and OpenAI's GPT-2 architecture outlined in [Language Models are Unsupervised Multitask Learners](https://d4mucfpruptmv.cloudfront.net/better-language-models/language-models_are_unsupervised_multitask_learners.pdf). 

Built using the [Burn deep-learning framework](https://burn.dev), it includes:

- ✅ **Custom Causal Self-Attention:** Custom Q/K/V combined projection with causal masking.
- ✅ **Embeddings:** Learned token and absolute positional embeddings.
- ✅ **Pre-norm Block Architecture:** Layer normalization applied before self-attention and MLP blocks.
- ✅ **GELU-activated MLP:** Standard GPT-2 feedforward network.
- ✅ **LM Head with Weight Tying:** Shares parameters with the token embedding table.
- ✅ **Pure-Rust BPE Tokenizer:** Fast `tiktoken-rs` wrapper using GPT-2's vocabulary.
- ✅ **Real-Time Word Streaming:** Outputs newly generated words instantly to the console.
- ✅ **GPU Model Weight Offloading:** Automatically drops weights from GPU VRAM immediately after inference finishes.
- ✅ **CUDA Execution:** GPU-accelerated by default using native Rust CUDA support (via CubeCL/cudarc) across Windows and Linux.
- ✅ **Single CLI Entry Point:** Reuses dependencies and compiles quickly into a single main executable.

---

## Model Sizes & Parameters

| Variant | Params | n_layer | n_head | n_embd |
|---------|--------|---------|--------|--------|
| `small` | 117 M  | 12      | 12     | 768    |
| `medium`| 345 M  | 24      | 16     | 1024   |
| `large` | 762 M  | 36      | 20     | 1280   |
| `xl`    | 1.5 B  | 48      | 25     | 1600   |

---

## Command-Line Interface (CLI) Usages

Everything compiles into a single, unified binary. Choose the operation mode using the flags:

### 1. Download Model Parameters (`--download`)
Downloads safetensors weights, model configurations, and vocabulary mapping for all variants from Hugging Face:
```bash
# Download small variant parameters into weights/small/
cargo run --release -- --download --size small --weights-dir weights

# Download all 4 variant parameters (small, medium, large, xl)
cargo run --release -- --download --size all --weights-dir weights
```
* **Arguments:**
  * `--size <small|medium|large|xl|all>`: The GPT-2 variant to fetch.
  * `--weights-dir <path>`: Directory to save downloaded files.
  * `--force`: Force download even if file already exists.

### 2. Generate Text / Run Inference (`--generate`)
Generates text autoregressively with real-time word streaming:
```bash
cargo run --release -- --generate --model weights/small/model.safetensors --prompt "The future of artificial intelligence is" --size small --max-new-tokens 100 --temperature 0.8 --top-k 50
```
* **Arguments:**
  * `--model <path>`: Path to model weights file.
  * `--size <small|medium|large|xl>`: Size configuration to construct.
  * `--prompt "<text>"`: Conditioning prompt text.
  * `--max-new-tokens <N>`: Maximum new tokens to produce.
  * `--temperature <F>`: Softmax temperature scaling (0 = greedy).
  * `--top-k <N>`: Top-K cutoff value (0 to disable).
  * `--top-p <F>`: Top-P (nucleus) value (0.0 to disable).
  * `--seed <N>`: Random seed for reproducibility.

### 3. Evaluate Perplexity (`--evaluate`)
Evaluates cross-entropy loss and perplexity on a test dataset:
```bash
cargo run --release -- --evaluate --model weights/small/model.safetensors --format safetensors --data data/input.txt --seq-len 128 --batch-size 4
```
* **Arguments:**
  * `--model <path>`: Path to weights file.
  * `--format <safetensors|compact>`: Weights format.
  * `--data <path>`: Path to validation dataset file.
  * `--seq-len <N>`: Sliding window sequence length.
  * `--batch-size <N>`: Evaluation batch size.

### 4. Pre-Train or Fine-Tune (`--train`)
Performs pre-training or fine-tuning from scratch:
```bash
cargo run --release -- --train --data data/input.txt --artifact-dir artifacts/ --size small --epochs 3 --batch-size 4 --seq-len 128 --lr 3e-4 --dropout 0.1
```
* **Arguments:**
  * `--data <path>`: Path to plain-text training file.
  * `--artifact-dir <path>`: Output directory for training checkpoints.
  * `--epochs <N>`: Number of training epochs.
  * `--lr <F>`: AdamW peak learning rate.
  * `--dropout <F>`: Dropout probability.

---

## Tech Stack

- **Deep Learning Framework:** [Burn](https://burn.dev) (v0.21)
- **GPU Engine:** WGPU (via Vulkan, Metal, or DX12)
- **Tokenizer:** `tiktoken-rs` (v0.5, BPE encoding)
- **Serialization:** `safetensors` (v0.4) & `memmap2` (v0.9)

---

## Citations & References

- **Burn Framework:**
  ```bibtex
  @misc{burn2024,
    title = {Burn: A Flexible and Modern Deep Learning Framework in Rust},
    howpublished = {\url{https://burn.dev/}},
    year = {2024}
  }
  ```
- **Attention Is All You Need:**
  - **arXiv:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
  - **Hugging Face Papers:** [https://huggingface.co/papers/1706.03762](https://huggingface.co/papers/1706.03762)
  ```bibtex
  @inproceedings{vaswani2017attention,
    author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
    title     = {Attention is all you need},
    booktitle = {Advances in neural information processing systems},
    pages     = {5998--6008},
    year      = {2017},
    url       = {https://arxiv.org/abs/1706.03762 or https://huggingface.co/papers/1706.03762}
  }
  ```
- **GPT-2 (Language Models are Unsupervised Multitask Learners):**
  ```bibtex
  @article{radford2019language,
    title   = {Language models are unsupervised multitask learners},
    author  = {Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
    journal = {OpenAI blog},
    volume  = {1},
    number  = {8},
    pages   = {9},
    year    = {2019},
    url     = {https://d4mucfpruptmv.cloudfront.net/better-language-models/language-models_are_unsupervised_multitask_learners.pdf}
  }
  ```
- **GPT-2 Rust (This repository):**
  ```bibtex
  @misc{gpt2rust2026,
    author       = {Muhammad Fiaz},
    title        = {GPT-2 Rust: A native Rust implementation of the GPT-2 transformer architecture built from scratch using the Burn deep-learning framework},
    howpublished = {\url{https://github.com/muhammad-fiaz/gpt-2-rust}},
    year         = {2026},
    note         = {Licensed under MIT License}
  }
  ```

---

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on code style, linting, and pull requests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
