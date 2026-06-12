//! Safetensors → Burn record loader.
//!
//! Maps the OpenAI GPT-2 safetensors key names to Burn `TensorData` so the
//! weights can be loaded into a [`Gpt2Model`] via manual record injection.
//!
//! ## Key mapping (OpenAI → Burn)
//!
//! | Safetensors key                       | Burn path                          |
//! |---------------------------------------|------------------------------------|
//! | `transformer.wte.weight`             | `wte.weight`                       |
//! | `transformer.wpe.weight`             | `wpe.weight`                       |
//! | `transformer.h.{i}.ln_1.weight`      | `blocks[i].ln_1.gamma`             |
//! | `transformer.h.{i}.ln_1.bias`        | `blocks[i].ln_1.beta`              |
//! | `transformer.h.{i}.attn.c_attn.weight`| `blocks[i].attn.c_attn.weight`    |
//! | `transformer.h.{i}.attn.c_attn.bias` | `blocks[i].attn.c_attn.bias`       |
//! | `transformer.h.{i}.attn.c_proj.weight`| `blocks[i].attn.c_proj.weight`    |
//! | `transformer.h.{i}.attn.c_proj.bias` | `blocks[i].attn.c_proj.bias`       |
//! | `transformer.h.{i}.ln_2.weight`      | `blocks[i].ln_2.gamma`             |
//! | `transformer.h.{i}.ln_2.bias`        | `blocks[i].ln_2.beta`              |
//! | `transformer.h.{i}.mlp.c_fc.weight`  | `blocks[i].mlp.c_fc.weight`        |
//! | `transformer.h.{i}.mlp.c_fc.bias`    | `blocks[i].mlp.c_fc.bias`          |
//! | `transformer.h.{i}.mlp.c_proj.weight`| `blocks[i].mlp.c_proj.weight`      |
//! | `transformer.h.{i}.mlp.c_proj.bias`  | `blocks[i].mlp.c_proj.bias`        |
//! | `transformer.ln_f.weight`            | `ln_f.gamma`                       |
//! | `transformer.ln_f.bias`              | `ln_f.beta`                        |
//! | `lm_head.weight`                     | `lm_head.weight`                   |

use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::module::Param;
use memmap2::Mmap;
use safetensors::SafeTensors;
use crate::model::Gpt2Model;

/// A flat map of tensor name → raw f32 data + shape, extracted from a
/// safetensors file.
pub type TensorMap = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Load all tensors from a `.safetensors` file into a [`TensorMap`].
///
/// Uses memory-mapped I/O for zero-copy access to large weight files.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<TensorMap> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Cannot open {:?}", path.as_ref()))?;

    // Safety: the file is read-only and we hold an exclusive `File` handle.
    let mmap = unsafe { Mmap::map(&file) }
        .context("Failed to mmap safetensors file")?;

    let tensors = SafeTensors::deserialize(&mmap)
        .context("Failed to deserialise safetensors header")?;

    let mut map = HashMap::new();

    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();

        // Convert raw bytes → f32 (safetensors stores BF16 or F32; we
        // always normalise to f32 for Burn's NdArray / Wgpu backends).
        let dtype = view.dtype();
        let data = view.data();

        let floats: Vec<f32> = match dtype {
            safetensors::Dtype::F32 => {
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            safetensors::Dtype::F16 => {
                data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        f16_to_f32(bits)
                    })
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        bf16_to_f32(bits)
                    })
                    .collect()
            }
            other => anyhow::bail!("Unsupported dtype {:?} for tensor '{}'", other, name),
        };

        map.insert(name.to_string(), (shape, floats));
    }

    Ok(map)
}

/// Build a flat key → `TensorData` map that Burn can consume.
///
/// The caller then sets each field on the model record by name.
pub fn build_tensor_data(tensor_map: &TensorMap) -> HashMap<String, TensorData> {
    tensor_map
        .iter()
        .map(|(name, (shape, data))| {
            let td = TensorData::new(data.clone(), shape.clone());
            (name.clone(), td)
        })
        .collect()
}

/// List all tensor keys in the file — useful for debugging mismatches.
pub fn list_keys(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let map = load_safetensors(path)?;
    let mut keys: Vec<String> = map.into_keys().collect();
    keys.sort();
    Ok(keys)
}

fn get_tensor<B: Backend, const D: usize>(
    tensor_map: &TensorMap,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>> {
    let (shape, data) = tensor_map
        .get(key)
        .with_context(|| format!("Tensor key not found in safetensors: '{}'", key))?;
    let shape_arr: [usize; D] = shape.clone().try_into().map_err(|_| {
        anyhow::anyhow!("Shape mismatch for {}: expected {} dims, got {:?}", key, D, shape)
    })?;
    Ok(Tensor::from_data(TensorData::new(data.clone(), shape_arr), device))
}

fn get_tensor_transposed<B: Backend>(
    tensor_map: &TensorMap,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    let (shape, data) = tensor_map
        .get(key)
        .with_context(|| format!("Tensor key not found in safetensors: '{}'", key))?;
    if shape.len() != 2 {
        anyhow::bail!("Expected 2D tensor for transposition, got {:?}", shape);
    }
    let shape_arr: [usize; 2] = [shape[0], shape[1]];
    let tensor: Tensor<B, 2> = Tensor::from_data(TensorData::new(data.clone(), shape_arr), device);
    Ok(tensor.transpose())
}

/// Load weights from a `.safetensors` file into the given `Gpt2Model`.
pub fn load_gpt2_weights<B: Backend>(
    model: Gpt2Model<B>,
    tensor_map: &TensorMap,
    device: &B::Device,
) -> Result<Gpt2Model<B>> {
    let model_to_load = model.clone();
    let mut model_record = model.into_record();

    let has_prefix = tensor_map.contains_key("transformer.wte.weight");
    let prefix = if has_prefix { "transformer." } else { "" };

    model_record.wte.weight = Param::from_tensor(get_tensor::<B, 2>(tensor_map, &format!("{}wte.weight", prefix), device)?);
    model_record.wpe.weight = Param::from_tensor(get_tensor::<B, 2>(tensor_map, &format!("{}wpe.weight", prefix), device)?);

    model_record.ln_f.gamma = Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}ln_f.weight", prefix), device)?);
    model_record.ln_f.beta = Some(Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}ln_f.bias", prefix), device)?));

    let lm_head_weight = if tensor_map.contains_key("lm_head.weight") {
        get_tensor_transposed::<B>(tensor_map, "lm_head.weight", device)?
    } else {
        let wte = get_tensor::<B, 2>(tensor_map, &format!("{}wte.weight", prefix), device)?;
        wte.transpose()
    };
    model_record.lm_head.weight = Param::from_tensor(lm_head_weight);
    model_record.lm_head.bias = None;

    let n_layer = model_record.blocks.len();
    for i in 0..n_layer {
        let block_prefix = format!("{}h.{}", prefix, i);
        let block_record = &mut model_record.blocks[i];

        block_record.ln_1.gamma = Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.ln_1.weight", block_prefix), device)?);
        block_record.ln_1.beta = Some(Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.ln_1.bias", block_prefix), device)?));

        block_record.attn.c_attn.weight = Param::from_tensor(get_tensor::<B, 2>(tensor_map, &format!("{}.attn.c_attn.weight", block_prefix), device)?);
        block_record.attn.c_attn.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.attn.c_attn.bias", block_prefix), device)?));

        block_record.attn.c_proj.weight = Param::from_tensor(get_tensor::<B, 2>(tensor_map, &format!("{}.attn.c_proj.weight", block_prefix), device)?);
        block_record.attn.c_proj.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.attn.c_proj.bias", block_prefix), device)?));

        block_record.ln_2.gamma = Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.ln_2.weight", block_prefix), device)?);
        block_record.ln_2.beta = Some(Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.ln_2.bias", block_prefix), device)?));

        block_record.mlp.c_fc.weight = Param::from_tensor(get_tensor::<B, 2>(tensor_map, &format!("{}.mlp.c_fc.weight", block_prefix), device)?);
        block_record.mlp.c_fc.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.mlp.c_fc.bias", block_prefix), device)?));

        block_record.mlp.c_proj.weight = Param::from_tensor(get_tensor::<B, 2>(tensor_map, &format!("{}.mlp.c_proj.weight", block_prefix), device)?);
        block_record.mlp.c_proj.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensor_map, &format!("{}.mlp.c_proj.bias", block_prefix), device)?));
    }
    Ok(model_to_load.load_record(model_record))
}

// dtype conversion helpers

/// Minimal f16 → f32 conversion without an external crate.
fn f16_to_f32(bits: u16) -> f32 {
    // IEEE 754 half-precision: 1 sign + 5 exp + 10 mantissa bits
    let sign: u32 = ((bits >> 15) as u32) << 31;
    let exp = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x3ff) as u32;

    if exp == 0 {
        // Subnormal
        let f = (mantissa as f32) / (1u32 << 24) as f32;
        if sign != 0 { -f } else { f }
    } else if exp == 0x1f {
        // Inf / NaN
        f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13))
    } else {
        f32::from_bits(sign | ((exp as u32 + 112) << 23) | (mantissa << 13))
    }
}

/// Minimal BF16 → f32 conversion.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0_f32);
    }

    #[test]
    fn test_f16_one() {
        // 1.0 in f16 = 0x3C00
        let v = f16_to_f32(0x3C00);
        assert!((v - 1.0_f32).abs() < 1e-5);
    }

    #[test]
    fn test_bf16_one() {
        // 1.0 in bf16 = 0x3F80
        let v = bf16_to_f32(0x3F80);
        assert!((v - 1.0_f32).abs() < 1e-5);
    }
}
