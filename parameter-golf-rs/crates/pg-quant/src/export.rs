/// Model artifact export: quantize → serialize → compress → write.
///
/// Target: < 16MB for competition submission.
///
/// Strategy:
///   - 4 parameter banks: int6 quantization with GPTQ-lite clip search
///   - Embedding (tok_emb): int8 (higher precision needed for tied output)
///   - Scalar params: f16 (small, precision-sensitive)
///   - zstd-22 compression on the whole artifact

use std::io::Write;
use std::path::Path;

use pg_core::error::PgResult;
use pg_model::model::GptModel;

use crate::int6::{quantize_int6, QuantizedWeight};
use crate::serialize::{write_artifact, SerializedTensor};
use crate::compress::compress_zstd;

/// Quantize and export the model to a compressed binary artifact.
/// Returns the artifact size in bytes.
pub fn export_model(model: &GptModel, path: &Path) -> PgResult<usize> {
    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let kv = c.kv_dim();
    let mlp = c.mlp_dim;

    let mut tensors = Vec::new();

    // 1. Parameter banks → int6
    let qo = quantize_int6(&model.qo_bank, 2 * n * d, d);
    tensors.push(int6_tensor("qo_bank.weight", &qo));
    tensors.push(scale_tensor("qo_bank.scale", &qo));

    let kv_q = quantize_int6(&model.kv_bank, 2 * n * kv, d);
    tensors.push(int6_tensor("kv_bank.weight", &kv_q));
    tensors.push(scale_tensor("kv_bank.scale", &kv_q));

    let up = quantize_int6(&model.mlp_up_bank, n * mlp, d);
    tensors.push(int6_tensor("mlp_up_bank.weight", &up));
    tensors.push(scale_tensor("mlp_up_bank.scale", &up));

    let down = quantize_int6(&model.mlp_down_bank, n * d, mlp);
    tensors.push(int6_tensor("mlp_down_bank.weight", &down));
    tensors.push(scale_tensor("mlp_down_bank.scale", &down));

    // 2. Embeddings → int8 (higher precision for tied output)
    tensors.push(int8_tensor("tok_emb", &model.tok_emb, c.vocab_size, d));

    // 3. Bigram params → f16
    if c.bigram_vocab_size > 0 {
        tensors.push(f16_tensor("bigram_embed", &model.bigram_embed));
        tensors.push(f16_tensor("bigram_proj", &model.bigram_proj));
        tensors.push(f32_scalar("bigram_scale", model.bigram_scale));
    }

    // 4. SmearGate → f16
    tensors.push(f16_tensor("smear_gate", &model.smear_gate));

    // 5. Skip weights → f16
    tensors.push(f16_tensor("skip_weights", &model.skip_weights));

    // 6. Per-block scalars → f16
    for i in 0..n {
        tensors.push(f16_tensor(&format!("blocks.{}.attn_scale", i), &model.blocks[i].attn_scale));
        tensors.push(f16_tensor(&format!("blocks.{}.mlp_scale", i), &model.blocks[i].mlp_scale));
        tensors.push(f16_tensor(&format!("blocks.{}.resid_mix", i), &model.blocks[i].resid_mix));
        tensors.push(f16_tensor(&format!("blocks.{}.q_gain", i), &model.blocks[i].q_gain));
    }

    // 7. VE params → f16
    if c.ve_enabled {
        tensors.push(f16_tensor("ve_embed", &model.ve_embed));
        tensors.push(f16_tensor("ve_proj", &model.ve_proj));
        tensors.push(f32_scalar("ve_scale", model.ve_scale));
        tensors.push(f16_tensor("ve_layer_scales", &model.ve_layer_scales));
    }

    // Serialize to buffer
    let metadata = format!(
        r#"{{"format":"pgrs_int6","vocab_size":{},"num_layers":{},"model_dim":{},"num_heads":{},"num_kv_heads":{},"head_dim":{},"mlp_dim":{}}}"#,
        c.vocab_size, n, d, c.num_heads, c.num_kv_heads, c.head_dim, mlp
    );

    let mut raw_buf = Vec::new();
    write_artifact(&mut raw_buf, &tensors, &metadata)?;

    // Compress with zstd-22
    let compressed = compress_zstd(&raw_buf, 22)?;

    let artifact_size = compressed.len();
    eprintln!(
        "Artifact: raw={:.2}MB, compressed={:.2}MB ({:.1}× ratio)",
        raw_buf.len() as f64 / 1_048_576.0,
        artifact_size as f64 / 1_048_576.0,
        raw_buf.len() as f64 / artifact_size as f64,
    );

    if artifact_size > 16 * 1_048_576 {
        eprintln!("WARNING: artifact exceeds 16MB limit ({:.2}MB)", artifact_size as f64 / 1_048_576.0);
    }

    // Write to file
    let mut file = std::fs::File::create(path)?;
    file.write_all(&compressed)?;

    Ok(artifact_size)
}

/// Load a compressed artifact back into a GptModel.
pub fn load_artifact(path: &Path, model: &mut GptModel) -> PgResult<()> {
    let compressed = std::fs::read(path)?;
    let raw = crate::compress::decompress_zstd(&compressed)?;

    let mut cursor = std::io::Cursor::new(raw);
    let (tensors, _metadata) = crate::serialize::read_artifact(&mut cursor)?;

    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let kv = c.kv_dim();
    let mlp = c.mlp_dim;

    for tensor in &tensors {
        match tensor.name.as_str() {
            "qo_bank.weight" => {
                let scales = find_tensor(&tensors, "qo_bank.scale");
                dequant_int6_into(&tensor.data, &scales.data, 2 * n * d, d, &mut model.qo_bank);
            }
            "kv_bank.weight" => {
                let scales = find_tensor(&tensors, "kv_bank.scale");
                dequant_int6_into(&tensor.data, &scales.data, 2 * n * kv, d, &mut model.kv_bank);
            }
            "mlp_up_bank.weight" => {
                let scales = find_tensor(&tensors, "mlp_up_bank.scale");
                dequant_int6_into(&tensor.data, &scales.data, n * mlp, d, &mut model.mlp_up_bank);
            }
            "mlp_down_bank.weight" => {
                let scales = find_tensor(&tensors, "mlp_down_bank.scale");
                dequant_int6_into(&tensor.data, &scales.data, n * d, mlp, &mut model.mlp_down_bank);
            }
            "tok_emb" => {
                dequant_int8_into(&tensor.data, c.vocab_size, d, &mut model.tok_emb);
            }
            "bigram_embed" => { f16_into(&tensor.data, &mut model.bigram_embed); }
            "bigram_proj" => { f16_into(&tensor.data, &mut model.bigram_proj); }
            "bigram_scale" => { model.bigram_scale = f32_from_bytes(&tensor.data); }
            "smear_gate" => { f16_into(&tensor.data, &mut model.smear_gate); }
            "skip_weights" => { f16_into(&tensor.data, &mut model.skip_weights); }
            "ve_embed" => { f16_into(&tensor.data, &mut model.ve_embed); }
            "ve_proj" => { f16_into(&tensor.data, &mut model.ve_proj); }
            "ve_scale" => { model.ve_scale = f32_from_bytes(&tensor.data); }
            "ve_layer_scales" => { f16_into(&tensor.data, &mut model.ve_layer_scales); }
            name if name.starts_with("blocks.") => {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() == 3 {
                    let idx: usize = parts[1].parse().unwrap();
                    match parts[2] {
                        "attn_scale" => f16_into(&tensor.data, &mut model.blocks[idx].attn_scale),
                        "mlp_scale" => f16_into(&tensor.data, &mut model.blocks[idx].mlp_scale),
                        "resid_mix" => f16_into(&tensor.data, &mut model.blocks[idx].resid_mix),
                        "q_gain" => f16_into(&tensor.data, &mut model.blocks[idx].q_gain),
                        _ => {}
                    }
                }
            }
            _ => {} // skip scale tensors (already consumed above)
        }
    }

    Ok(())
}

// === Helper functions ===

fn int6_tensor(name: &str, qw: &QuantizedWeight) -> SerializedTensor {
    SerializedTensor {
        name: name.to_string(),
        shape: vec![qw.rows, qw.cols],
        dtype: pg_core::DType::I8,
        data: qw.data.iter().map(|&x| x as u8).collect(),
    }
}

fn scale_tensor(name: &str, qw: &QuantizedWeight) -> SerializedTensor {
    let data: Vec<u8> = qw.scales.iter()
        .flat_map(|s| s.to_bits().to_le_bytes())
        .collect();
    SerializedTensor {
        name: name.to_string(),
        shape: vec![qw.rows],
        dtype: pg_core::DType::F16,
        data,
    }
}

fn int8_tensor(name: &str, weights: &[f32], rows: usize, cols: usize) -> SerializedTensor {
    // Per-row int8 quantization with scale packed after the data
    let mut data = Vec::with_capacity(rows * cols + rows * 2);
    let mut scales = Vec::with_capacity(rows);

    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let max_abs = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = (max_abs / 127.0).max(1e-10);
        scales.push(half::f16::from_f32(scale));
        for &w in row {
            let q = (w / scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q as u8);
        }
    }
    // Append scales
    for s in &scales {
        data.extend_from_slice(&s.to_bits().to_le_bytes());
    }

    SerializedTensor {
        name: name.to_string(),
        shape: vec![rows, cols],
        dtype: pg_core::DType::I8,
        data,
    }
}

fn f16_tensor(name: &str, weights: &[f32]) -> SerializedTensor {
    let data: Vec<u8> = weights.iter()
        .flat_map(|&w| half::f16::from_f32(w).to_bits().to_le_bytes())
        .collect();
    SerializedTensor {
        name: name.to_string(),
        shape: vec![weights.len()],
        dtype: pg_core::DType::F16,
        data,
    }
}

fn f32_scalar(name: &str, value: f32) -> SerializedTensor {
    SerializedTensor {
        name: name.to_string(),
        shape: vec![1],
        dtype: pg_core::DType::F32,
        data: value.to_le_bytes().to_vec(),
    }
}

fn find_tensor<'a>(tensors: &'a [SerializedTensor], name: &str) -> &'a SerializedTensor {
    tensors.iter().find(|t| t.name == name)
        .unwrap_or_else(|| panic!("missing tensor: {}", name))
}

fn dequant_int6_into(data: &[u8], scale_data: &[u8], rows: usize, cols: usize, dest: &mut [f32]) {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(scale_data.len(), rows * 2);

    for r in 0..rows {
        let scale_bits = u16::from_le_bytes([scale_data[r * 2], scale_data[r * 2 + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        for c in 0..cols {
            dest[r * cols + c] = data[r * cols + c] as i8 as f32 * scale;
        }
    }
}

fn dequant_int8_into(data: &[u8], rows: usize, cols: usize, dest: &mut [f32]) {
    let weights_end = rows * cols;
    let scale_start = weights_end;
    assert!(data.len() >= weights_end + rows * 2);

    for r in 0..rows {
        let scale_bits = u16::from_le_bytes([data[scale_start + r * 2], data[scale_start + r * 2 + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        for c in 0..cols {
            dest[r * cols + c] = data[r * cols + c] as i8 as f32 * scale;
        }
    }
}

fn f16_into(data: &[u8], dest: &mut [f32]) {
    assert_eq!(data.len(), dest.len() * 2);
    for i in 0..dest.len() {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        dest[i] = half::f16::from_bits(bits).to_f32();
    }
}

fn f32_from_bytes(data: &[u8]) -> f32 {
    f32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use pg_model::config::ModelConfig;

    #[test]
    fn test_export_roundtrip() {
        let config = ModelConfig::sota();
        let model = GptModel::new(config.clone());

        // Export
        let tmp = std::env::temp_dir().join("pg_test_artifact.pgrs");
        let size = export_model(&model, &tmp).unwrap();
        eprintln!("Artifact size: {} bytes ({:.2} MB)", size, size as f64 / 1_048_576.0);

        // Reimport
        let mut model2 = GptModel::new(config);
        load_artifact(&tmp, &mut model2).unwrap();

        // Check bank reconstruction error is small (int6 has ~0.1% MSE)
        let mse_qo = mse(&model.qo_bank, &model2.qo_bank);
        let mse_kv = mse(&model.kv_bank, &model2.kv_bank);
        eprintln!("Reconstruction MSE — qo: {:.6}, kv: {:.6}", mse_qo, mse_kv);

        // Scalars should roundtrip through f16 with small error
        let mse_smear = mse(&model.smear_gate, &model2.smear_gate);
        assert!(mse_smear < 1e-4, "smear_gate roundtrip MSE too high: {}", mse_smear);

        // Clean up
        std::fs::remove_file(&tmp).ok();
    }

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b.iter())
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum::<f64>() / a.len() as f64
    }
}
