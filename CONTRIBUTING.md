# Contributing to GPT-2 Rust 🦀🔥

We welcome contributions from the community! Whether you want to fix a bug, improve performance, add a new feature, or improve documentation, this guide will help you get started.

## How to Contribute

### 1. Open an Issue
Before making any major changes, please open an issue to discuss your proposed updates. This helps align our design goals and avoids redundant work.

### 2. Fork & Branch
- Fork the repository on GitHub.
- Create a new branch naming it appropriately (e.g. `feat/kv-cache`, `fix/causal-mask`).

### 3. Coding Guidelines
- **Rust Version:** The project uses stable Rust (Edition 2024). Make sure you run with a modern toolchain.
- **Formatting:** Format your code using `cargo fmt` before submitting a PR.
- **Linting:** Ensure that `cargo clippy` runs without warnings.
- **Testing:** Verify all unit tests pass with `cargo test`.

### 4. Submitting a Pull Request
- Submit your pull request targeting the `main` branch.
- Provide a clear description of the problem your changes address, the rationale behind your decisions, and evidence that it works correctly.
- Be sure to update documentation or tests if applicable.

---

## Technical Overview

- **Deep Learning Framework:** We use [Burn 0.21](https://burn.dev/) for tensor operations, neural network modules, config structures, and autodiff.
- **GPU Acceleration:** High-performance native NVIDIA GPU computing is orchestrated via `burn-cuda` (CubeCL/cudarc). Keep device mapping robust and GPU performance-friendly.
- **No Python Wrapper:** All parts of this model are implemented natively in Rust. Keep the dependency list lean and avoid unnecessary binds.

Thank you for contributing to GPT-2 Rust!
