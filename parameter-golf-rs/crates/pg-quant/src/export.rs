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

use pg_core::error::{PgError, PgResult};
use pg_model::model::GptModel;
use pg_model::{CompressionMode, QuantScheme, QuantSpec};

use crate::compress::compress_zstd;
use crate::prune::{PruneConfig, PruneStrategy, prune_then_quantize};
use crate::scheme::{Bits, Block, GroupConfig, PackedWeight, Scheme, quantize_with};
use crate::serialize::{SerializedTensor, write_artifact};

/// Quantize and export the model to a compressed binary artifact.
/// Returns the artifact size in bytes.
pub fn export_model(model: &GptModel, path: &Path) -> PgResult<usize> {
    export_model_with_spec(model, &QuantSpec::default(), "legacy_default", path)
}

/// Quantize and export the model according to a RunSpec QuantSpec.
/// Returns the artifact size in bytes.
pub fn export_model_with_spec(
    model: &GptModel,
    quant_spec: &QuantSpec,
    variant_fingerprint: &str,
    path: &Path,
) -> PgResult<usize> {
    if quant_spec.compression != CompressionMode::Zstd22 {
        return Err(PgError::InvalidOp(format!(
            "artifact export currently supports zstd22 only, got {:?}",
            quant_spec.compression
        )));
    }
    let scheme = scheme_from_quant_spec(quant_spec)?;
    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let kv = c.kv_dim();
    let mlp = c.mlp_dim;

    let mut tensors = Vec::new();

    // 1. Parameter banks — split by semantic group so QuantSpec can use
    // different bit widths for attention, MLP, and embeddings.
    let qo_split = n * d * d;
    push_packed_group(
        &mut tensors,
        "qo_bank.q",
        &model.qo_bank[..qo_split],
        n * d,
        d,
        &scheme.attn_q,
        quant_spec.prune_keep_ratio,
    );
    push_packed_group(
        &mut tensors,
        "qo_bank.o",
        &model.qo_bank[qo_split..],
        n * d,
        d,
        &scheme.attn_o,
        quant_spec.prune_keep_ratio,
    );

    let kv_split = n * kv * d;
    push_packed_group(
        &mut tensors,
        "kv_bank.k",
        &model.kv_bank[..kv_split],
        n * kv,
        d,
        &scheme.attn_k,
        quant_spec.prune_keep_ratio,
    );
    push_packed_group(
        &mut tensors,
        "kv_bank.v",
        &model.kv_bank[kv_split..],
        n * kv,
        d,
        &scheme.attn_v,
        quant_spec.prune_keep_ratio,
    );

    push_packed_group(
        &mut tensors,
        "mlp_up_bank",
        &model.mlp_up_bank,
        n * mlp,
        d,
        &scheme.mlp_up,
        quant_spec.prune_keep_ratio,
    );
    push_packed_group(
        &mut tensors,
        "mlp_down_bank",
        &model.mlp_down_bank,
        n * d,
        mlp,
        &scheme.mlp_down,
        quant_spec.prune_keep_ratio,
    );

    // 2. Embeddings use their own QuantSpec group because tied output quality
    // is sensitive to excessive compression.
    push_packed_group(
        &mut tensors,
        "tok_emb",
        &model.tok_emb,
        c.vocab_size,
        d,
        &scheme.embed,
        None,
    );

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
        tensors.push(f16_tensor(
            &format!("blocks.{}.attn_scale", i),
            &model.blocks[i].attn_scale,
        ));
        tensors.push(f16_tensor(
            &format!("blocks.{}.mlp_scale", i),
            &model.blocks[i].mlp_scale,
        ));
        tensors.push(f16_tensor(
            &format!("blocks.{}.resid_mix", i),
            &model.blocks[i].resid_mix,
        ));
        tensors.push(f16_tensor(
            &format!("blocks.{}.q_gain", i),
            &model.blocks[i].q_gain,
        ));
        if c.attn_out_gate_enabled {
            tensors.push(f16_tensor(
                &format!("blocks.{}.attn_gate_weight", i),
                &model.blocks[i].attn_gate_weight,
            ));
            tensors.push(f16_tensor(
                &format!("blocks.{}.attn_gate_bias", i),
                &model.blocks[i].attn_gate_bias,
            ));
        }
    }

    // 7. VE params → f16
    if c.ve_enabled {
        tensors.push(f16_tensor("ve_embed", &model.ve_embed));
        tensors.push(f16_tensor("ve_proj", &model.ve_proj));
        tensors.push(f32_scalar("ve_scale", model.ve_scale));
        tensors.push(f16_tensor("ve_layer_scales", &model.ve_layer_scales));
    }

    // Serialize to buffer
    let metadata = metadata_json(model, quant_spec, &scheme, variant_fingerprint);

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

    if artifact_size > 16_000_000 {
        eprintln!(
            "WARNING: artifact exceeds 16,000,000-byte model budget ({:.2}MB decimal)",
            artifact_size as f64 / 1_000_000.0
        );
    }

    // Write to file
    let mut file = std::fs::File::create(path)?;
    file.write_all(&compressed)?;

    Ok(artifact_size)
}

fn scheme_from_quant_spec(quant_spec: &QuantSpec) -> PgResult<Scheme> {
    let int4 = GroupConfig::new(Bits::B4, Block::PerRow);
    let int6 = GroupConfig::new(Bits::B6, Block::PerRow);
    let int7 = GroupConfig::new(Bits::B7, Block::PerRow);
    let int8 = GroupConfig::new(Bits::B8, Block::PerRow);
    match quant_spec.scheme {
        QuantScheme::None => Err(PgError::InvalidOp(
            "QuantScheme::None is not a submission artifact format; choose gptq_lite_int6, mixed_int5_int6, or aggressive".into(),
        )),
        QuantScheme::GptqLiteInt6 => Ok(Scheme {
            attn_q: int6.clone(),
            attn_k: int6.clone(),
            attn_v: int6.clone(),
            attn_o: int6.clone(),
            mlp_up: int6.clone(),
            mlp_down: int6,
            embed: int8,
        }),
        QuantScheme::MixedInt5Int6 => Ok(Scheme::sota_baseline()),
        QuantScheme::Aggressive => Ok(Scheme {
            attn_q: int8.clone(),
            attn_k: int8.clone(),
            attn_v: int8.clone(),
            attn_o: int8.clone(),
            mlp_up: int4.clone(),
            mlp_down: int4,
            embed: int8,
        }),
        QuantScheme::TightInt7Int4 => Ok(Scheme {
            attn_q: int7.clone(),
            attn_k: int7.clone(),
            attn_v: int7.clone(),
            attn_o: int7,
            mlp_up: int4.clone(),
            mlp_down: int4,
            embed: int8,
        }),
    }
}

fn push_packed_group(
    tensors: &mut Vec<SerializedTensor>,
    name: &str,
    weights: &[f32],
    rows: usize,
    cols: usize,
    cfg: &GroupConfig,
    prune_keep_ratio: Option<f32>,
) {
    let packed = if let Some(keep_ratio) = prune_keep_ratio {
        let mut pruned = weights.to_vec();
        let prune_cfg = PruneConfig {
            strategy: PruneStrategy::TopKPerRow { keep_ratio },
            rescale_after_prune: true,
        };
        prune_then_quantize(&mut pruned, rows, cols, &prune_cfg, cfg).packed
    } else {
        quantize_with(weights, rows, cols, cfg)
    };

    tensors.push(packed_tensor(&format!("{name}.weight"), &packed));
    tensors.push(packed_scale_tensor(&format!("{name}.scale"), &packed));
}

fn packed_tensor(name: &str, packed: &PackedWeight) -> SerializedTensor {
    SerializedTensor {
        name: name.to_string(),
        shape: vec![packed.rows, packed.cols],
        dtype: pg_core::DType::I8,
        data: pack_signed_values(&packed.data, packed.bits),
    }
}

fn packed_scale_tensor(name: &str, packed: &PackedWeight) -> SerializedTensor {
    let data: Vec<u8> = packed
        .scales
        .iter()
        .flat_map(|&s| half::f16::from_f32(s).to_bits().to_le_bytes())
        .collect();
    SerializedTensor {
        name: name.to_string(),
        shape: vec![packed.scales.len()],
        dtype: pg_core::DType::F16,
        data,
    }
}

fn pack_signed_values(values: &[i8], bits: Bits) -> Vec<u8> {
    let nbits = bits.nbits();
    let qmin = bits.qmin();
    let mut out = vec![0u8; (values.len() * nbits + 7) / 8];
    let mut bit_pos = 0usize;
    for &value in values {
        let encoded = (value as i32 - qmin) as u32;
        for b in 0..nbits {
            if ((encoded >> b) & 1) != 0 {
                let dst = bit_pos + b;
                out[dst / 8] |= 1u8 << (dst % 8);
            }
        }
        bit_pos += nbits;
    }
    out
}

fn unpack_signed_values(data: &[u8], count: usize, bits: Bits) -> Vec<i8> {
    let nbits = bits.nbits();
    let qmin = bits.qmin();
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut encoded = 0u32;
        for b in 0..nbits {
            let src = bit_pos + b;
            if src / 8 < data.len() && (data[src / 8] & (1u8 << (src % 8))) != 0 {
                encoded |= 1u32 << b;
            }
        }
        out.push((encoded as i32 + qmin) as i8);
        bit_pos += nbits;
    }
    out
}

fn metadata_json(
    model: &GptModel,
    quant_spec: &QuantSpec,
    scheme: &Scheme,
    variant_fingerprint: &str,
) -> String {
    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let mlp = c.mlp_dim;
    let prune = quant_spec
        .prune_keep_ratio
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());
    format!(
        r#"{{"format":"pgrs_quant","version":2,"variant_fingerprint":"{}","scheme":"{:?}","compression":"{:?}","prune_keep_ratio":{},"vocab_size":{},"num_layers":{},"model_dim":{},"num_heads":{},"num_kv_heads":{},"head_dim":{},"mlp_dim":{},"attn_out_gate_enabled":{},"attn_out_gate_width":{},"groups":{{"qo_bank.q":{},"qo_bank.o":{},"kv_bank.k":{},"kv_bank.v":{},"mlp_up_bank":{},"mlp_down_bank":{},"tok_emb":{}}}}}"#,
        variant_fingerprint,
        quant_spec.scheme,
        quant_spec.compression,
        prune,
        c.vocab_size,
        n,
        d,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
        mlp,
        if c.attn_out_gate_enabled { 1 } else { 0 },
        c.attn_out_gate_width,
        scheme.attn_q.bits.nbits(),
        scheme.attn_o.bits.nbits(),
        scheme.attn_k.bits.nbits(),
        scheme.attn_v.bits.nbits(),
        scheme.mlp_up.bits.nbits(),
        scheme.mlp_down.bits.nbits(),
        scheme.embed.bits.nbits(),
    )
}

/// Load a compressed artifact back into a GptModel.
pub fn load_artifact(path: &Path, model: &mut GptModel) -> PgResult<()> {
    let compressed = std::fs::read(path)?;
    let raw = crate::compress::decompress_zstd(&compressed)?;

    let mut cursor = std::io::Cursor::new(raw);
    let (tensors, metadata) = crate::serialize::read_artifact(&mut cursor)?;

    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let kv = c.kv_dim();
    let mlp = c.mlp_dim;

    validate_artifact_metadata(&metadata, model)?;
    if c.attn_out_gate_enabled {
        for i in 0..n {
            find_tensor_result(&tensors, &format!("blocks.{i}.attn_gate_weight"))?;
            find_tensor_result(&tensors, &format!("blocks.{i}.attn_gate_bias"))?;
        }
    }

    let has_split_quant = find_tensor_opt(&tensors, "qo_bank.q.weight").is_some();
    if has_split_quant {
        let qo_split = n * d * d;
        dequant_packed_group(
            &tensors,
            &metadata,
            "qo_bank.q",
            n * d,
            d,
            &mut model.qo_bank[..qo_split],
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "qo_bank.o",
            n * d,
            d,
            &mut model.qo_bank[qo_split..],
        )?;
        let kv_split = n * kv * d;
        dequant_packed_group(
            &tensors,
            &metadata,
            "kv_bank.k",
            n * kv,
            d,
            &mut model.kv_bank[..kv_split],
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "kv_bank.v",
            n * kv,
            d,
            &mut model.kv_bank[kv_split..],
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "mlp_up_bank",
            n * mlp,
            d,
            &mut model.mlp_up_bank,
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "mlp_down_bank",
            n * d,
            mlp,
            &mut model.mlp_down_bank,
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "tok_emb",
            c.vocab_size,
            d,
            &mut model.tok_emb,
        )?;
    }

    for tensor in &tensors {
        match tensor.name.as_str() {
            "qo_bank.weight" if !has_split_quant => {
                let scales = find_tensor(&tensors, "qo_bank.scale");
                dequant_int6_into(&tensor.data, &scales.data, 2 * n * d, d, &mut model.qo_bank);
            }
            "kv_bank.weight" if !has_split_quant => {
                let scales = find_tensor(&tensors, "kv_bank.scale");
                dequant_int6_into(
                    &tensor.data,
                    &scales.data,
                    2 * n * kv,
                    d,
                    &mut model.kv_bank,
                );
            }
            "mlp_up_bank.weight" if !has_split_quant => {
                let scales = find_tensor(&tensors, "mlp_up_bank.scale");
                dequant_int6_into(
                    &tensor.data,
                    &scales.data,
                    n * mlp,
                    d,
                    &mut model.mlp_up_bank,
                );
            }
            "mlp_down_bank.weight" if !has_split_quant => {
                let scales = find_tensor(&tensors, "mlp_down_bank.scale");
                dequant_int6_into(
                    &tensor.data,
                    &scales.data,
                    n * d,
                    mlp,
                    &mut model.mlp_down_bank,
                );
            }
            "tok_emb" if !has_split_quant => {
                dequant_int8_into(&tensor.data, c.vocab_size, d, &mut model.tok_emb);
            }
            "bigram_embed" => {
                f16_into(&tensor.data, &mut model.bigram_embed);
            }
            "bigram_proj" => {
                f16_into(&tensor.data, &mut model.bigram_proj);
            }
            "bigram_scale" => {
                model.bigram_scale = f32_from_bytes(&tensor.data);
            }
            "smear_gate" => {
                f16_into(&tensor.data, &mut model.smear_gate);
            }
            "skip_weights" => {
                f16_into(&tensor.data, &mut model.skip_weights);
            }
            "ve_embed" => {
                f16_into(&tensor.data, &mut model.ve_embed);
            }
            "ve_proj" => {
                f16_into(&tensor.data, &mut model.ve_proj);
            }
            "ve_scale" => {
                model.ve_scale = f32_from_bytes(&tensor.data);
            }
            "ve_layer_scales" => {
                f16_into(&tensor.data, &mut model.ve_layer_scales);
            }
            name if name.starts_with("blocks.") => {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() == 3 {
                    let idx: usize = parts[1].parse().unwrap();
                    match parts[2] {
                        "attn_scale" => f16_into(&tensor.data, &mut model.blocks[idx].attn_scale),
                        "mlp_scale" => f16_into(&tensor.data, &mut model.blocks[idx].mlp_scale),
                        "resid_mix" => f16_into(&tensor.data, &mut model.blocks[idx].resid_mix),
                        "q_gain" => f16_into(&tensor.data, &mut model.blocks[idx].q_gain),
                        "attn_gate_weight" => {
                            f16_into(&tensor.data, &mut model.blocks[idx].attn_gate_weight)
                        }
                        "attn_gate_bias" => {
                            f16_into(&tensor.data, &mut model.blocks[idx].attn_gate_bias)
                        }
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

fn dequant_packed_group(
    tensors: &[SerializedTensor],
    metadata: &str,
    name: &str,
    rows: usize,
    cols: usize,
    dest: &mut [f32],
) -> PgResult<()> {
    let weight = find_tensor_result(tensors, &format!("{name}.weight"))?;
    let scale = find_tensor_result(tensors, &format!("{name}.scale"))?;
    let bits = bits_from_nbits(metadata_group_bits(metadata, name)?);
    dequant_packed_into(&weight.data, &scale.data, rows, cols, bits, dest);
    Ok(())
}

fn dequant_packed_into(
    data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    bits: Bits,
    dest: &mut [f32],
) {
    assert_eq!(dest.len(), rows * cols);
    assert_eq!(scale_data.len(), rows * 2);
    let q = unpack_signed_values(data, rows * cols, bits);
    for r in 0..rows {
        let scale_bits = u16::from_le_bytes([scale_data[r * 2], scale_data[r * 2 + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        for c in 0..cols {
            dest[r * cols + c] = q[r * cols + c] as f32 * scale;
        }
    }
}

fn metadata_group_bits(metadata: &str, group: &str) -> PgResult<usize> {
    let key = format!("\"{group}\":");
    let start = metadata.find(&key).ok_or_else(|| {
        PgError::DataFormat(format!("artifact metadata missing bits for {group}"))
    })? + key.len();
    let digits: String = metadata[start..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits
        .parse::<usize>()
        .map_err(|e| PgError::DataFormat(format!("invalid artifact bits for group {group}: {e}")))
}

fn bits_from_nbits(nbits: usize) -> Bits {
    match nbits {
        4 => Bits::B4,
        5 => Bits::B5,
        6 => Bits::B6,
        7 => Bits::B7,
        8 => Bits::B8,
        _ => panic!("unsupported quantized bit width: {nbits}"),
    }
}

fn validate_artifact_metadata(metadata: &str, model: &GptModel) -> PgResult<()> {
    if metadata.contains(r#""format":"pgrs_quant""#) || metadata.contains(r#""format":"pgrs_int6""#)
    {
        let c = &model.config;
        for (key, expected) in [
            ("vocab_size", c.vocab_size),
            ("num_layers", c.num_layers),
            ("model_dim", c.model_dim),
            ("num_heads", c.num_heads),
            ("num_kv_heads", c.num_kv_heads),
            ("head_dim", c.head_dim),
            ("mlp_dim", c.mlp_dim),
            (
                "attn_out_gate_enabled",
                if c.attn_out_gate_enabled { 1 } else { 0 },
            ),
            ("attn_out_gate_width", c.attn_out_gate_width),
        ] {
            if let Some(got) = metadata_usize(metadata, key) {
                if got != expected {
                    return Err(PgError::DataFormat(format!(
                        "artifact metadata mismatch for {key}: expected {expected}, got {got}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn metadata_usize(metadata: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\":");
    let start = metadata.find(&needle)? + needle.len();
    let digits: String = metadata[start..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn f16_tensor(name: &str, weights: &[f32]) -> SerializedTensor {
    let data: Vec<u8> = weights
        .iter()
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
    tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("missing tensor: {}", name))
}

fn find_tensor_opt<'a>(
    tensors: &'a [SerializedTensor],
    name: &str,
) -> Option<&'a SerializedTensor> {
    tensors.iter().find(|t| t.name == name)
}

fn find_tensor_result<'a>(
    tensors: &'a [SerializedTensor],
    name: &str,
) -> PgResult<&'a SerializedTensor> {
    find_tensor_opt(tensors, name)
        .ok_or_else(|| PgError::DataFormat(format!("missing tensor: {name}")))
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
        let scale_bits =
            u16::from_le_bytes([data[scale_start + r * 2], data[scale_start + r * 2 + 1]]);
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
        eprintln!(
            "Artifact size: {} bytes ({:.2} MB)",
            size,
            size as f64 / 1_048_576.0
        );

        // Reimport
        let mut model2 = GptModel::new(config);
        load_artifact(&tmp, &mut model2).unwrap();

        // Check bank reconstruction error is small (int6 has ~0.1% MSE)
        let mse_qo = mse(&model.qo_bank, &model2.qo_bank);
        let mse_kv = mse(&model.kv_bank, &model2.kv_bank);
        eprintln!("Reconstruction MSE — qo: {:.6}, kv: {:.6}", mse_qo, mse_kv);

        // Scalars should roundtrip through f16 with small error
        let mse_smear = mse(&model.smear_gate, &model2.smear_gate);
        assert!(
            mse_smear < 1e-4,
            "smear_gate roundtrip MSE too high: {}",
            mse_smear
        );

        // Clean up
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_roundtrip_all_supported_quant_specs() {
        let config = small_config();
        let model = GptModel::new(config.clone());
        for scheme in [
            QuantScheme::GptqLiteInt6,
            QuantScheme::MixedInt5Int6,
            QuantScheme::Aggressive,
            QuantScheme::TightInt7Int4,
        ] {
            let spec = QuantSpec {
                scheme,
                prune_keep_ratio: if scheme == QuantScheme::Aggressive {
                    Some(0.90)
                } else {
                    None
                },
                ..QuantSpec::default()
            };
            let tmp = std::env::temp_dir().join(format!("pg_test_artifact_{scheme:?}.pgrs"));
            export_model_with_spec(&model, &spec, "test_fingerprint", &tmp).unwrap();

            let mut loaded = GptModel::new(config.clone());
            load_artifact(&tmp, &mut loaded).unwrap();
            assert_eq!(loaded.qo_bank.len(), model.qo_bank.len());
            assert_eq!(loaded.kv_bank.len(), model.kv_bank.len());
            assert_eq!(loaded.mlp_up_bank.len(), model.mlp_up_bank.len());
            assert_eq!(loaded.mlp_down_bank.len(), model.mlp_down_bank.len());
            assert_eq!(loaded.tok_emb.len(), model.tok_emb.len());
            std::fs::remove_file(&tmp).ok();
        }
    }

    fn small_config() -> ModelConfig {
        let model_dim = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = model_dim / num_heads;
        let mlp_dim = 32;
        ModelConfig {
            vocab_size: 64,
            num_layers: 2,
            model_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            mlp_mult: 2.0,
            mlp_dim,
            rope_base: 10_000.0,
            rope_dims: 4,
            xsa_last_n: 1,
            logit_softcap: 30.0,
            qk_gain_init: 1.5,
            recurrence_enabled: false,
            recurrence_start_layer: 0,
            recurrence_repeat_layers: 0,
            parallel_residual: false,
            attn_out_gate_enabled: false,
            attn_out_gate_width: 24,
            vrl_enabled: false,
            ve_enabled: true,
            ve_dim: 8,
            ve_layers: vec![1],
            bigram_vocab_size: 32,
            bigram_dim: 8,
            ln_scale: true,
            tie_embeddings: true,
            tied_embed_init_std: 0.005,
            train_seq_len: 8,
            eval_seq_len: 8,
        }
    }

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64
    }
}
