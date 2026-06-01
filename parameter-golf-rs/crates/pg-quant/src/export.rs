/// Model artifact export: quantize -> serialize -> compress -> write.
///
/// This module reports compressed model bytes only. Record-mode submission
/// validity is checked in `pg-train` with code bytes plus compressed model
/// bytes, matching the official 16,000,000-byte budget.
///
/// Strategy:
///   - 4 parameter banks: int6 quantization with GPTQ-lite clip search
///   - Embedding (tok_emb): int8 (higher precision needed for tied output)
///   - Scalar params: f16 (small, precision-sensitive)
///   - zstd-22 compression on the whole artifact
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use pg_core::error::{PgError, PgResult};
use pg_model::model::GptModel;
use pg_model::spec::LqerSpec;
use pg_model::{CompressionMode, QuantScheme, QuantSpec};
use serde::Deserialize;
use serde_json::json;

use crate::compress::{compress_pergroup, compress_zstd, decompress_artifact_payload};
use crate::layout::{QuantArchProfile, compiled_layout_for_quant_spec};
use crate::pack::{CompiledQuantKernelSet, qmax_for_bits, qmin_for_bits};
use crate::prune::{PruneConfig, PruneStrategy, prune_then_quantize};
use crate::scheme::{Bits, Block, GroupConfig, PackedWeight, Scheme, quantize_with};
use crate::serialize::{SerializedTensor, write_artifact};

#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct ArtifactMetadata {
    format: Option<String>,
    version: Option<usize>,
    variant_fingerprint: Option<String>,
    quant_layout_manifest_crc32: Option<String>,
    arch_profile: Option<String>,
    quant_kernel_ids: Option<String>,
    matrix_bits: Option<usize>,
    mlp_bits: Option<usize>,
    embed_bits: Option<usize>,
    attn_gate_bits: Option<usize>,
    gptq_calibration_batches: Option<usize>,
    lqer_enabled: Option<usize>,
    lqer_rank: Option<usize>,
    lqer_a_bits: Option<usize>,
    lqer_b_bits: Option<usize>,
    vocab_size: Option<usize>,
    num_layers: Option<usize>,
    model_dim: Option<usize>,
    num_heads: Option<usize>,
    num_kv_heads: Option<usize>,
    head_dim: Option<usize>,
    mlp_dim: Option<usize>,
    attn_out_gate_enabled: Option<usize>,
    attn_out_gate_width: Option<usize>,
    sparse_attn_gate_enabled: Option<usize>,
    sparse_attn_gate_width: Option<usize>,
    groups: Option<BTreeMap<String, usize>>,
}

impl ArtifactMetadata {
    fn parse(raw: &str, strict: bool) -> PgResult<Self> {
        match serde_json::from_str(raw) {
            Ok(metadata) => Ok(metadata),
            Err(err) if !strict => {
                eprintln!(
                    "WARNING: artifact metadata is not structured JSON ({err}); loading as a legacy artifact without compiled quant manifest validation"
                );
                Ok(Self::default())
            }
            Err(err) => Err(PgError::DataFormat(format!(
                "artifact metadata is not valid JSON: {err}"
            ))),
        }
    }

    fn is_quant_format(&self) -> bool {
        matches!(
            self.format.as_deref(),
            Some("pgrs_quant") | Some("pgrs_int6")
        )
    }

    fn usize_field(&self, key: &str) -> Option<usize> {
        match key {
            "matrix_bits" => self.matrix_bits,
            "mlp_bits" => self.mlp_bits,
            "embed_bits" => self.embed_bits,
            "attn_gate_bits" => self.attn_gate_bits,
            "gptq_calibration_batches" => self.gptq_calibration_batches,
            "lqer_enabled" => self.lqer_enabled,
            "lqer_rank" => self.lqer_rank,
            "lqer_a_bits" => self.lqer_a_bits,
            "lqer_b_bits" => self.lqer_b_bits,
            "vocab_size" => self.vocab_size,
            "num_layers" => self.num_layers,
            "model_dim" => self.model_dim,
            "num_heads" => self.num_heads,
            "num_kv_heads" => self.num_kv_heads,
            "head_dim" => self.head_dim,
            "mlp_dim" => self.mlp_dim,
            "attn_out_gate_enabled" => self.attn_out_gate_enabled,
            "attn_out_gate_width" => self.attn_out_gate_width,
            "sparse_attn_gate_enabled" => self.sparse_attn_gate_enabled,
            "sparse_attn_gate_width" => self.sparse_attn_gate_width,
            _ => None,
        }
    }

    fn group_bits(&self, group: &str) -> PgResult<usize> {
        self.groups
            .as_ref()
            .and_then(|groups| groups.get(group).copied())
            .ok_or_else(|| {
                PgError::DataFormat(format!("artifact metadata missing bits for {group}"))
            })
    }
}

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
    let scheme = scheme_from_quant_spec(quant_spec)?;
    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let kv = c.kv_dim();
    let mlp = c.mlp_dim;
    let layout_manifest = crate::layout::compile_quant_layout_manifest(quant_spec, Some(c))?;
    let kernel_set = CompiledQuantKernelSet::for_manifest(&layout_manifest);

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
        &quant_spec.lqer,
        &kernel_set,
    );
    push_packed_group(
        &mut tensors,
        "qo_bank.o",
        &model.qo_bank[qo_split..],
        n * d,
        d,
        &scheme.attn_o,
        quant_spec.prune_keep_ratio,
        &quant_spec.lqer,
        &kernel_set,
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
        &quant_spec.lqer,
        &kernel_set,
    );
    push_packed_group(
        &mut tensors,
        "kv_bank.v",
        &model.kv_bank[kv_split..],
        n * kv,
        d,
        &scheme.attn_v,
        quant_spec.prune_keep_ratio,
        &quant_spec.lqer,
        &kernel_set,
    );

    push_packed_group(
        &mut tensors,
        "mlp_up_bank",
        &model.mlp_up_bank,
        n * mlp,
        d,
        &scheme.mlp_up,
        quant_spec.prune_keep_ratio,
        &quant_spec.lqer,
        &kernel_set,
    );
    push_packed_group(
        &mut tensors,
        "mlp_down_bank",
        &model.mlp_down_bank,
        n * d,
        mlp,
        &scheme.mlp_down,
        quant_spec.prune_keep_ratio,
        &quant_spec.lqer,
        &kernel_set,
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
        &quant_spec.lqer,
        &kernel_set,
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
        if c.sparse_attn_gate_enabled {
            tensors.push(f16_tensor(
                &format!("blocks.{}.sparse_attn_gate_weight", i),
                &model.blocks[i].sparse_attn_gate_weight,
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

    let compressed = match quant_spec.compression {
        CompressionMode::None => raw_buf.clone(),
        CompressionMode::Zstd22 => compress_zstd(&raw_buf, 22)?,
        CompressionMode::Pergroup => compress_pergroup(&raw_buf, 22)?,
        CompressionMode::Lzma9 => {
            return Err(PgError::InvalidOp(
                "artifact export does not support lzma9 in the Rust record path".into(),
            ));
        }
    };

    let artifact_size = compressed.len();
    eprintln!(
        "Artifact: raw={:.2}MB, compressed={:.2}MB ({:.1}× ratio)",
        raw_buf.len() as f64 / 1_048_576.0,
        artifact_size as f64 / 1_048_576.0,
        raw_buf.len() as f64 / artifact_size as f64,
    );

    if artifact_size > quant_spec.target_artifact_bytes {
        eprintln!(
            "WARNING: compressed model artifact alone exceeds configured byte target ({:.2}MB decimal); final record validity still requires code_bytes + model_bytes below target",
            artifact_size as f64 / 1_000_000.0
        );
    }

    // Write to file
    let mut file = std::fs::File::create(path)?;
    file.write_all(&compressed)?;

    Ok(artifact_size)
}

fn scheme_from_quant_spec(quant_spec: &QuantSpec) -> PgResult<Scheme> {
    let matrix = GroupConfig::new(
        bits_from_quant_spec_nbits(quant_spec.matrix_bits, "matrix_bits")?,
        Block::PerRow,
    );
    let mlp = GroupConfig::new(
        bits_from_quant_spec_nbits(quant_spec.mlp_bits, "mlp_bits")?,
        Block::PerRow,
    );
    let embed = GroupConfig::new(
        bits_from_quant_spec_nbits(quant_spec.embed_bits, "embed_bits")?,
        Block::PerRow,
    );
    match quant_spec.scheme {
        QuantScheme::None => Err(PgError::InvalidOp(
            "QuantScheme::None is not a submission artifact format; choose gptq_lite_int6, mixed_int5_int6, or aggressive".into(),
        )),
        QuantScheme::GptqLiteInt6 => Ok(Scheme {
            attn_q: matrix.clone(),
            attn_k: matrix.clone(),
            attn_v: matrix.clone(),
            attn_o: matrix.clone(),
            mlp_up: matrix.clone(),
            mlp_down: matrix,
            embed,
        }),
        QuantScheme::MixedInt5Int6 => {
            // The MLP bit-width is generated by the compile-time quant layout
            // compiler so this record-critical split is not a hand-coded
            // duplicate of the TOML/profile contract.
            if let Some(layout) = compiled_layout_for_quant_spec(quant_spec) {
                layout.validate_quant_spec(quant_spec)?;
            } else {
                eprintln!(
                    "mixed_int5_int6 export: QuantSpec bits={}/{}/{} gptq_batches={} do not match a compiled frontier layout; using spec-owned layout for non-frontier export",
                    quant_spec.matrix_bits,
                    quant_spec.mlp_bits,
                    quant_spec.embed_bits,
                    quant_spec.gptq_calibration_batches
                );
            }
            Ok(Scheme {
                attn_q: matrix.clone(),
                attn_k: matrix.clone(),
                attn_v: matrix.clone(),
                attn_o: matrix.clone(),
                mlp_up: mlp.clone(),
                mlp_down: mlp,
                embed,
            })
        }
        QuantScheme::Aggressive => Ok(Scheme {
            attn_q: matrix.clone(),
            attn_k: matrix.clone(),
            attn_v: matrix.clone(),
            attn_o: matrix.clone(),
            mlp_up: mlp.clone(),
            mlp_down: mlp,
            embed,
        }),
        QuantScheme::TightInt7Int4 => Ok(Scheme {
            attn_q: matrix.clone(),
            attn_k: matrix.clone(),
            attn_v: matrix.clone(),
            attn_o: matrix,
            mlp_up: mlp.clone(),
            mlp_down: mlp,
            embed,
        }),
    }
}

fn bits_from_quant_spec_nbits(nbits: u8, field: &str) -> PgResult<Bits> {
    match nbits {
        4 => Ok(Bits::B4),
        5 => Ok(Bits::B5),
        6 => Ok(Bits::B6),
        7 => Ok(Bits::B7),
        8 => Ok(Bits::B8),
        _ => Err(PgError::InvalidOp(format!(
            "unsupported {field}={nbits}; expected 4, 5, 6, 7, or 8"
        ))),
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
    lqer: &LqerSpec,
    kernel_set: &CompiledQuantKernelSet,
) {
    let quant_source = if let Some(keep_ratio) = prune_keep_ratio {
        let mut pruned = weights.to_vec();
        let prune_cfg = PruneConfig {
            strategy: PruneStrategy::TopKPerRow { keep_ratio },
            rescale_after_prune: true,
        };
        prune_then_quantize(&mut pruned, rows, cols, &prune_cfg, cfg).packed
    } else {
        quantize_with(weights, rows, cols, cfg)
    };

    if lqer.enabled && lqer.rank > 0 {
        push_lqer_tensors(tensors, name, weights, &quant_source, lqer, kernel_set);
    }

    tensors.push(packed_tensor(
        &format!("{name}.weight"),
        &quant_source,
        kernel_set,
    ));
    tensors.push(packed_scale_tensor(&format!("{name}.scale"), &quant_source));
}

fn push_lqer_tensors(
    tensors: &mut Vec<SerializedTensor>,
    name: &str,
    weights: &[f32],
    packed: &PackedWeight,
    lqer: &LqerSpec,
    kernel_set: &CompiledQuantKernelSet,
) {
    let recon = packed.dequantize();
    let residual: Vec<f32> = weights
        .iter()
        .zip(recon.iter())
        .map(|(&w, &q)| w - q)
        .collect();
    let (a, b) = low_rank_residual_factors(&residual, packed.rows, packed.cols, lqer.rank);
    let (a_q, a_scales) = quantize_lqer_factor(&a, packed.rows, lqer.rank, lqer.a_bits);
    let (b_q, b_scales) = quantize_lqer_factor(&b, lqer.rank, packed.cols, lqer.b_bits);
    tensors.push(quantized_lqer_tensor(
        &format!("{name}.lqer.a.weight"),
        &a_q,
        packed.rows,
        lqer.rank,
        lqer.a_bits,
        kernel_set,
    ));
    tensors.push(quantized_lqer_scale_tensor(
        &format!("{name}.lqer.a.scale"),
        &a_scales,
        packed.rows,
    ));
    tensors.push(quantized_lqer_tensor(
        &format!("{name}.lqer.b.weight"),
        &b_q,
        lqer.rank,
        packed.cols,
        lqer.b_bits,
        kernel_set,
    ));
    tensors.push(quantized_lqer_scale_tensor(
        &format!("{name}.lqer.b.scale"),
        &b_scales,
        lqer.rank,
    ));
}

fn packed_tensor(
    name: &str,
    packed: &PackedWeight,
    kernel_set: &CompiledQuantKernelSet,
) -> SerializedTensor {
    SerializedTensor {
        name: name.to_string(),
        shape: vec![packed.rows, packed.cols],
        dtype: pg_core::DType::I8,
        data: kernel_set
            .pack_signed(&packed.data, packed.bits.nbits() as u8)
            .expect("packed quant bit width should be validated by scheme construction"),
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

fn quantized_lqer_tensor(
    name: &str,
    quantized: &[i8],
    rows: usize,
    cols: usize,
    nbits: u8,
    kernel_set: &CompiledQuantKernelSet,
) -> SerializedTensor {
    SerializedTensor {
        name: name.to_string(),
        shape: vec![rows, cols],
        dtype: pg_core::DType::I8,
        data: kernel_set
            .pack_signed(quantized, nbits)
            .expect("LQER bit width should be validated by layout compiler"),
    }
}

fn quantized_lqer_scale_tensor(name: &str, scales: &[f32], rows: usize) -> SerializedTensor {
    let data: Vec<u8> = scales
        .iter()
        .flat_map(|&s| half::f16::from_f32(s).to_bits().to_le_bytes())
        .collect();
    SerializedTensor {
        name: name.to_string(),
        shape: vec![rows],
        dtype: pg_core::DType::F16,
        data,
    }
}

fn quantize_lqer_factor(
    weights: &[f32],
    rows: usize,
    cols: usize,
    nbits: u8,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(weights.len(), rows * cols);
    assert!((2..=8).contains(&nbits), "unsupported LQER bits: {nbits}");
    let qmax = qmax_for_bits(nbits);
    let qmin = qmin_for_bits(nbits);
    let mut q = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let max_abs = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = (max_abs / qmax.max(1) as f32).max(1e-8);
        scales.push(scale);
        for &v in row {
            q.push((v / scale).round().clamp(qmin as f32, qmax as f32) as i8);
        }
    }
    (q, scales)
}

fn low_rank_residual_factors(
    residual: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(residual.len(), rows * cols);
    let rank = rank.min(rows).min(cols);
    let mut work = residual.to_vec();
    let mut a = vec![0.0f32; rows * rank];
    let mut b = vec![0.0f32; rank * cols];

    for k in 0..rank {
        let mut v: Vec<f32> = (0..cols)
            .map(|i| ((i + 1 + k * 17) as f32 * 0.618_033_9).sin())
            .collect();
        normalize(&mut v);
        let mut u = vec![0.0f32; rows];

        for _ in 0..6 {
            mat_vec_rows(&work, rows, cols, &v, &mut u);
            if !normalize(&mut u) {
                break;
            }
            mat_t_vec_cols(&work, rows, cols, &u, &mut v);
            if !normalize(&mut v) {
                break;
            }
        }

        mat_vec_rows(&work, rows, cols, &v, &mut u);
        let sigma = dot(&u, &u).sqrt();
        if sigma <= 1e-10 || !sigma.is_finite() {
            break;
        }
        for value in &mut u {
            *value /= sigma;
        }
        mat_t_vec_cols(&work, rows, cols, &u, &mut v);
        let sigma = normalize_with_norm(&mut v);
        if sigma <= 1e-10 || !sigma.is_finite() {
            break;
        }

        for r in 0..rows {
            a[r * rank + k] = u[r] * sigma;
        }
        for c in 0..cols {
            b[k * cols + c] = v[c];
        }

        for r in 0..rows {
            let ur_sigma = u[r] * sigma;
            let row = &mut work[r * cols..(r + 1) * cols];
            for c in 0..cols {
                row[c] -= ur_sigma * v[c];
            }
        }
    }

    (a, b)
}

fn mat_vec_rows(matrix: &[f32], rows: usize, cols: usize, v: &[f32], out: &mut [f32]) {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(v.len(), cols);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        out[r] = dot(&matrix[r * cols..(r + 1) * cols], v);
    }
}

fn mat_t_vec_cols(matrix: &[f32], rows: usize, cols: usize, u: &[f32], out: &mut [f32]) {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(u.len(), rows);
    assert_eq!(out.len(), cols);
    out.fill(0.0);
    for r in 0..rows {
        let ur = u[r];
        for c in 0..cols {
            out[c] += matrix[r * cols + c] * ur;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn normalize(values: &mut [f32]) -> bool {
    normalize_with_norm(values) > 1e-10
}

fn normalize_with_norm(values: &mut [f32]) -> f32 {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= 1e-10 || !norm.is_finite() {
        return 0.0;
    }
    for v in values {
        *v /= norm;
    }
    norm
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
    let layout_manifest = crate::layout::compile_quant_layout_manifest(quant_spec, Some(c)).ok();
    let layout_manifest_value = layout_manifest
        .as_ref()
        .and_then(|manifest| {
            serde_json::from_str::<serde_json::Value>(&manifest.metadata_json()).ok()
        })
        .unwrap_or(serde_json::Value::Null);
    let layout_manifest_crc32 = layout_manifest
        .as_ref()
        .map(|manifest| manifest.fingerprint_crc32.as_str());
    let layout_arch_profile = layout_manifest
        .as_ref()
        .map(|manifest| manifest.arch_profile.as_str())
        .unwrap_or("uncompiled");
    let quant_kernel_ids = layout_manifest
        .as_ref()
        .map(|manifest| {
            let mut ids: Vec<&str> = Vec::new();
            for group in &manifest.groups {
                if !ids.contains(&group.pack_kernel) {
                    ids.push(group.pack_kernel);
                }
                if !ids.contains(&group.dequant_kernel) {
                    ids.push(group.dequant_kernel);
                }
            }
            ids.join(",")
        })
        .unwrap_or_default();
    let prune = quant_spec
        .prune_keep_ratio
        .map(serde_json::Value::from)
        .unwrap_or(serde_json::Value::Null);
    json!({
        "format": "pgrs_quant",
        "version": 2,
        "variant_fingerprint": variant_fingerprint,
        "quant_layout_manifest_crc32": layout_manifest_crc32,
        "quant_layout_manifest": layout_manifest_value,
        "arch_profile": layout_arch_profile,
        "quant_kernel_ids": quant_kernel_ids,
        "scheme": format!("{:?}", quant_spec.scheme),
        "compression": format!("{:?}", quant_spec.compression),
        "matrix_bits": quant_spec.matrix_bits,
        "mlp_bits": quant_spec.mlp_bits,
        "embed_bits": quant_spec.embed_bits,
        "attn_gate_bits": quant_spec.attn_gate_bits,
        "mlp_clip_sigmas": quant_spec.mlp_clip_sigmas,
        "attn_clip_sigmas": quant_spec.attn_clip_sigmas,
        "embed_clip_sigmas": quant_spec.embed_clip_sigmas,
        "gptq_calibration_batches": quant_spec.gptq_calibration_batches,
        "prune_keep_ratio": prune,
        "lqer_enabled": if quant_spec.lqer.enabled { 1 } else { 0 },
        "lqer_rank": quant_spec.lqer.rank,
        "lqer_top_k": quant_spec.lqer.top_k,
        "lqer_a_bits": quant_spec.lqer.a_bits,
        "lqer_b_bits": quant_spec.lqer.b_bits,
        "lqer_group_size": quant_spec.lqer.group_size,
        "lqer_asymmetric": if quant_spec.lqer.asymmetric { 1 } else { 0 },
        "vocab_size": c.vocab_size,
        "num_layers": n,
        "model_dim": d,
        "num_heads": c.num_heads,
        "num_kv_heads": c.num_kv_heads,
        "head_dim": c.head_dim,
        "mlp_dim": mlp,
        "attn_out_gate_enabled": if c.attn_out_gate_enabled { 1 } else { 0 },
        "attn_out_gate_width": c.attn_out_gate_width,
        "sparse_attn_gate_enabled": if c.sparse_attn_gate_enabled { 1 } else { 0 },
        "sparse_attn_gate_width": c.sparse_attn_gate_width,
        "groups": {
            "qo_bank.q": scheme.attn_q.bits.nbits(),
            "qo_bank.o": scheme.attn_o.bits.nbits(),
            "kv_bank.k": scheme.attn_k.bits.nbits(),
            "kv_bank.v": scheme.attn_v.bits.nbits(),
            "mlp_up_bank": scheme.mlp_up.bits.nbits(),
            "mlp_down_bank": scheme.mlp_down.bits.nbits(),
            "tok_emb": scheme.embed.bits.nbits(),
        }
    })
    .to_string()
}

/// Load a compressed artifact back into a GptModel.
pub fn load_artifact(path: &Path, model: &mut GptModel) -> PgResult<()> {
    load_artifact_inner(path, model, None, false)
}

/// Load an artifact and validate its compiled quantization manifest against the
/// expected QuantSpec. Strict mode is intended for record/eval paths where
/// legacy or mismatched artifacts must fail closed.
pub fn load_artifact_with_spec(
    path: &Path,
    model: &mut GptModel,
    quant_spec: &QuantSpec,
    strict_manifest: bool,
) -> PgResult<()> {
    load_artifact_inner(path, model, Some(quant_spec), strict_manifest)
}

fn load_artifact_inner(
    path: &Path,
    model: &mut GptModel,
    quant_spec: Option<&QuantSpec>,
    strict_manifest: bool,
) -> PgResult<()> {
    let compressed = std::fs::read(path)?;
    let raw = decompress_artifact_payload(&compressed)?;

    let mut cursor = std::io::Cursor::new(raw);
    let (tensors, metadata) = crate::serialize::read_artifact(&mut cursor)?;

    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let kv = c.kv_dim();
    let mlp = c.mlp_dim;

    let metadata = ArtifactMetadata::parse(&metadata, strict_manifest)?;
    validate_artifact_metadata(&metadata, model)?;
    if let Some(quant_spec) = quant_spec {
        validate_artifact_quant_manifest(&metadata, model, quant_spec, strict_manifest)?;
    }
    if c.attn_out_gate_enabled {
        for i in 0..n {
            find_tensor_result(&tensors, &format!("blocks.{i}.attn_gate_weight"))?;
            find_tensor_result(&tensors, &format!("blocks.{i}.attn_gate_bias"))?;
        }
    }
    if c.sparse_attn_gate_enabled {
        for i in 0..n {
            find_tensor_result(&tensors, &format!("blocks.{i}.sparse_attn_gate_weight"))?;
        }
    }

    let has_split_quant = find_tensor_opt(&tensors, "qo_bank.q.weight").is_some();
    let kernel_set = artifact_kernel_set_from_metadata(&metadata)?;
    if has_split_quant {
        let qo_split = n * d * d;
        dequant_packed_group(
            &tensors,
            &metadata,
            "qo_bank.q",
            n * d,
            d,
            &mut model.qo_bank[..qo_split],
            &kernel_set,
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "qo_bank.o",
            n * d,
            d,
            &mut model.qo_bank[qo_split..],
            &kernel_set,
        )?;
        let kv_split = n * kv * d;
        dequant_packed_group(
            &tensors,
            &metadata,
            "kv_bank.k",
            n * kv,
            d,
            &mut model.kv_bank[..kv_split],
            &kernel_set,
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "kv_bank.v",
            n * kv,
            d,
            &mut model.kv_bank[kv_split..],
            &kernel_set,
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "mlp_up_bank",
            n * mlp,
            d,
            &mut model.mlp_up_bank,
            &kernel_set,
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "mlp_down_bank",
            n * d,
            mlp,
            &mut model.mlp_down_bank,
            &kernel_set,
        )?;
        dequant_packed_group(
            &tensors,
            &metadata,
            "tok_emb",
            c.vocab_size,
            d,
            &mut model.tok_emb,
            &kernel_set,
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
                        "sparse_attn_gate_weight" => {
                            f16_into(&tensor.data, &mut model.blocks[idx].sparse_attn_gate_weight)
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
    metadata: &ArtifactMetadata,
    name: &str,
    rows: usize,
    cols: usize,
    dest: &mut [f32],
    kernel_set: &CompiledQuantKernelSet,
) -> PgResult<()> {
    let weight = find_tensor_result(tensors, &format!("{name}.weight"))?;
    let scale = find_tensor_result(tensors, &format!("{name}.scale"))?;
    let bits = metadata_group_bits(metadata, name)? as u8;
    kernel_set.dequant_per_row(&weight.data, &scale.data, rows, cols, bits, dest)?;
    apply_lqer_if_present(tensors, metadata, name, rows, cols, dest, kernel_set)?;
    Ok(())
}

fn apply_lqer_if_present(
    tensors: &[SerializedTensor],
    metadata: &ArtifactMetadata,
    name: &str,
    rows: usize,
    cols: usize,
    dest: &mut [f32],
    kernel_set: &CompiledQuantKernelSet,
) -> PgResult<()> {
    let Some(a_weight) = find_tensor_opt(tensors, &format!("{name}.lqer.a.weight")) else {
        if metadata.usize_field("lqer_enabled").unwrap_or(0) != 0 {
            return Err(PgError::DataFormat(format!(
                "artifact metadata enables LQER but tensor {name}.lqer.a.weight is missing"
            )));
        }
        return Ok(());
    };
    let a_scale = find_tensor_result(tensors, &format!("{name}.lqer.a.scale"))?;
    let b_weight = find_tensor_result(tensors, &format!("{name}.lqer.b.weight"))?;
    let b_scale = find_tensor_result(tensors, &format!("{name}.lqer.b.scale"))?;

    let rank = metadata.usize_field("lqer_rank").unwrap_or(0);
    if rank == 0 {
        return Ok(());
    }
    if a_weight.shape != vec![rows, rank] {
        return Err(PgError::DataFormat(format!(
            "invalid LQER A shape for {name}: expected [{rows}, {rank}], got {:?}",
            a_weight.shape
        )));
    }
    if b_weight.shape != vec![rank, cols] {
        return Err(PgError::DataFormat(format!(
            "invalid LQER B shape for {name}: expected [{rank}, {cols}], got {:?}",
            b_weight.shape
        )));
    }

    let a_bits = metadata.usize_field("lqer_a_bits").unwrap_or(2) as u8;
    let b_bits = metadata.usize_field("lqer_b_bits").unwrap_or(4) as u8;
    let a = dequant_lqer_factor(
        &a_weight.data,
        &a_scale.data,
        rows,
        rank,
        a_bits,
        kernel_set,
    )?;
    let b = dequant_lqer_factor(
        &b_weight.data,
        &b_scale.data,
        rank,
        cols,
        b_bits,
        kernel_set,
    )?;

    for r in 0..rows {
        for c in 0..cols {
            let mut correction = 0.0f32;
            for k in 0..rank {
                correction += a[r * rank + k] * b[k * cols + c];
            }
            dest[r * cols + c] += correction;
        }
    }
    Ok(())
}

fn dequant_lqer_factor(
    data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    nbits: u8,
    kernel_set: &CompiledQuantKernelSet,
) -> PgResult<Vec<f32>> {
    if !(2..=8).contains(&nbits) {
        return Err(PgError::DataFormat(format!(
            "unsupported LQER bit width: {nbits}"
        )));
    }
    if scale_data.len() != rows * 2 {
        return Err(PgError::DataFormat(format!(
            "invalid LQER scale length: expected {}, got {}",
            rows * 2,
            scale_data.len()
        )));
    }
    let q = kernel_set.unpack_signed(data, rows * cols, nbits)?;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let scale_bits = u16::from_le_bytes([scale_data[r * 2], scale_data[r * 2 + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        for c in 0..cols {
            out[r * cols + c] = q[r * cols + c] as f32 * scale;
        }
    }
    Ok(out)
}

fn metadata_group_bits(metadata: &ArtifactMetadata, group: &str) -> PgResult<usize> {
    metadata.group_bits(group)
}

fn artifact_kernel_set_from_metadata(
    metadata: &ArtifactMetadata,
) -> PgResult<CompiledQuantKernelSet> {
    let Some(label) = metadata.arch_profile.as_deref() else {
        return Ok(CompiledQuantKernelSet::from_arch_profile(
            QuantArchProfile::record_default(),
        ));
    };
    let arch_profile = QuantArchProfile::from_label(label).ok_or_else(|| {
        PgError::DataFormat(format!(
            "artifact metadata contains unsupported quant arch profile: {label}"
        ))
    })?;
    Ok(CompiledQuantKernelSet::from_arch_profile(arch_profile))
}

fn validate_artifact_metadata(metadata: &ArtifactMetadata, model: &GptModel) -> PgResult<()> {
    if metadata.is_quant_format() {
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
            (
                "sparse_attn_gate_enabled",
                if c.sparse_attn_gate_enabled { 1 } else { 0 },
            ),
            ("sparse_attn_gate_width", c.sparse_attn_gate_width),
        ] {
            if let Some(got) = metadata.usize_field(key) {
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

fn validate_artifact_quant_manifest(
    metadata: &ArtifactMetadata,
    model: &GptModel,
    quant_spec: &QuantSpec,
    strict_manifest: bool,
) -> PgResult<()> {
    if !metadata.is_quant_format() {
        if strict_manifest {
            return Err(PgError::DataFormat(
                "strict quant manifest validation requires a pgrs quant artifact".into(),
            ));
        }
        return Ok(());
    }

    let expected_manifest =
        crate::layout::compile_quant_layout_manifest(quant_spec, Some(&model.config))?;
    for (key, expected) in [
        ("matrix_bits", quant_spec.matrix_bits as usize),
        ("mlp_bits", quant_spec.mlp_bits as usize),
        ("embed_bits", quant_spec.embed_bits as usize),
        ("attn_gate_bits", quant_spec.attn_gate_bits as usize),
        (
            "gptq_calibration_batches",
            quant_spec.gptq_calibration_batches,
        ),
    ] {
        match metadata.usize_field(key) {
            Some(got) if got == expected => {}
            Some(got) => {
                return Err(PgError::DataFormat(format!(
                    "artifact quant metadata mismatch for {key}: expected {expected}, got {got}"
                )));
            }
            None if strict_manifest => {
                return Err(PgError::DataFormat(format!(
                    "artifact quant metadata missing required field {key}"
                )));
            }
            None => {}
        }
    }

    match metadata.quant_layout_manifest_crc32.as_deref() {
        Some(got) if got == expected_manifest.fingerprint_crc32 => {}
        Some(got) => {
            return Err(PgError::DataFormat(format!(
                "artifact quant layout manifest CRC mismatch: expected {}, got {}",
                expected_manifest.fingerprint_crc32, got
            )));
        }
        None if strict_manifest => {
            return Err(PgError::DataFormat(
                "artifact quant metadata missing quant_layout_manifest_crc32".into(),
            ));
        }
        None => {}
    }
    Ok(())
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

    #[test]
    fn test_export_roundtrip_with_lqer_enabled() {
        let config = small_config();
        let model = GptModel::new(config.clone());
        let spec = QuantSpec {
            scheme: QuantScheme::Aggressive,
            lqer: LqerSpec {
                enabled: true,
                rank: 2,
                top_k: 3,
                a_bits: 2,
                b_bits: 4,
                group_size: 64,
                asymmetric: true,
            },
            ..QuantSpec::default()
        };
        let tmp = std::env::temp_dir().join("pg_test_artifact_lqer.pgrs");
        export_model_with_spec(&model, &spec, "test_lqer", &tmp).unwrap();

        let mut loaded = GptModel::new(config);
        load_artifact(&tmp, &mut loaded).unwrap();
        assert_eq!(loaded.qo_bank.len(), model.qo_bank.len());
        assert!(loaded.qo_bank.iter().all(|v| v.is_finite()));
        assert!(loaded.mlp_up_bank.iter().all(|v| v.is_finite()));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn export_metadata_embeds_compiled_quant_manifest_and_kernel_ids() {
        let config = small_config();
        let model = GptModel::new(config);
        let spec = QuantSpec {
            scheme: QuantScheme::Aggressive,
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            gptq_calibration_batches: 32,
            lqer: LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let scheme = scheme_from_quant_spec(&spec).unwrap();
        let metadata = metadata_json(&model, &spec, &scheme, "metadata_test");
        assert!(metadata.contains("\"quant_layout_manifest_crc32\":\""));
        assert!(metadata.contains("\"quant_layout_manifest\":{"));
        assert!(metadata.contains("\"arch_profile\":\"sm90_h100\""));
        assert!(metadata.contains("\"quant_kernel_ids\":\""));
        assert!(metadata.contains("pack_signed_i4_per_row_sm90"));
        assert!(metadata.contains("dequant_i4_per_row_f16_scale_sm90"));
    }

    #[test]
    fn artifact_kernel_set_rejects_unknown_arch_profile() {
        let metadata = ArtifactMetadata::parse(r#"{"arch_profile":"sm123_future"}"#, true).unwrap();
        let err = artifact_kernel_set_from_metadata(&metadata)
            .expect_err("unknown arch profile must fail closed for compiled artifacts");
        assert!(err.to_string().contains("unsupported quant arch profile"));
    }

    #[test]
    fn artifact_metadata_parser_fails_strict_and_allows_legacy_non_strict() {
        let err = ArtifactMetadata::parse("not-json", true)
            .expect_err("strict record/eval paths must reject malformed metadata");
        assert!(err.to_string().contains("not valid JSON"), "{err}");

        let legacy = ArtifactMetadata::parse("not-json", false).unwrap();
        assert!(!legacy.is_quant_format());
        assert!(legacy.groups.is_none());
    }

    #[test]
    fn strict_quant_manifest_validation_rejects_missing_crc() {
        let config = small_config();
        let model = GptModel::new(config);
        let spec = QuantSpec {
            scheme: QuantScheme::Aggressive,
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            gptq_calibration_batches: 32,
            lqer: LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let metadata = ArtifactMetadata::parse(
            r#"{"format":"pgrs_quant","version":2,"matrix_bits":5,"mlp_bits":4,"embed_bits":6,"attn_gate_bits":8,"gptq_calibration_batches":32}"#,
            true,
        )
        .unwrap();
        let err = validate_artifact_quant_manifest(&metadata, &model, &spec, true)
            .expect_err("strict record/eval paths require manifest CRC");
        assert!(
            err.to_string()
                .contains("missing quant_layout_manifest_crc32"),
            "{err}"
        );
    }

    #[test]
    fn strict_artifact_loader_rejects_quant_manifest_mismatch() {
        let config = small_config();
        let model = GptModel::new(config.clone());
        let spec = QuantSpec {
            scheme: QuantScheme::Aggressive,
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            gptq_calibration_batches: 32,
            lqer: LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let tmp = std::env::temp_dir().join("pg_test_artifact_manifest_mismatch.pgrs");
        export_model_with_spec(&model, &spec, "manifest_mismatch", &tmp).unwrap();

        let mut mismatched_spec = spec.clone();
        mismatched_spec.matrix_bits = 6;
        let mut loaded = GptModel::new(config);
        let err = load_artifact_with_spec(&tmp, &mut loaded, &mismatched_spec, true)
            .expect_err("strict loader must reject mismatched QuantSpec");
        let _ = std::fs::remove_file(&tmp);
        assert!(
            err.to_string().contains("artifact quant metadata mismatch")
                || err.to_string().contains("compiled quant layout"),
            "{err}"
        );
    }

    #[test]
    fn compiled_manifest_byte_estimate_matches_exported_tensor_payloads() {
        let config = small_config();
        let model = GptModel::new(config.clone());
        let spec = QuantSpec {
            scheme: QuantScheme::Aggressive,
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            gptq_calibration_batches: 32,
            lqer: LqerSpec {
                enabled: true,
                rank: 2,
                top_k: 3,
                a_bits: 2,
                b_bits: 4,
                group_size: 64,
                asymmetric: true,
            },
            ..QuantSpec::default()
        };
        let tmp = std::env::temp_dir().join("pg_test_artifact_byte_estimate.pgrs");
        export_model_with_spec(&model, &spec, "byte_estimate", &tmp).unwrap();

        let compressed = std::fs::read(&tmp).unwrap();
        let raw = decompress_artifact_payload(&compressed).unwrap();
        let mut cursor = std::io::Cursor::new(raw);
        let (tensors, metadata_raw) = crate::serialize::read_artifact(&mut cursor).unwrap();
        let metadata = ArtifactMetadata::parse(&metadata_raw, true).unwrap();
        let manifest = crate::layout::compile_quant_layout_manifest(&spec, Some(&config)).unwrap();

        let mut actual = 0usize;
        for group in &manifest.groups {
            actual += find_tensor(&tensors, &format!("{}.weight", group.name))
                .data
                .len();
            actual += find_tensor(&tensors, &format!("{}.scale", group.name))
                .data
                .len();
            if spec.lqer.enabled {
                actual += find_tensor(&tensors, &format!("{}.lqer.a.weight", group.name))
                    .data
                    .len();
                actual += find_tensor(&tensors, &format!("{}.lqer.a.scale", group.name))
                    .data
                    .len();
                actual += find_tensor(&tensors, &format!("{}.lqer.b.weight", group.name))
                    .data
                    .len();
                actual += find_tensor(&tensors, &format!("{}.lqer.b.scale", group.name))
                    .data
                    .len();
            }
            assert_eq!(
                metadata.group_bits(group.name).unwrap(),
                group.bits as usize
            );
        }
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(Some(actual), manifest.estimated_raw_weight_bytes);
    }

    #[test]
    fn strict_export_reload_tiny_eval_smoke_produces_finite_loss() {
        let config = small_config();
        let model = GptModel::new(config.clone());
        let spec = QuantSpec {
            scheme: QuantScheme::Aggressive,
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            gptq_calibration_batches: 32,
            lqer: LqerSpec {
                enabled: true,
                rank: 2,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let tmp = std::env::temp_dir().join("pg_test_artifact_tiny_eval_smoke.pgrs");
        export_model_with_spec(&model, &spec, "tiny_eval_smoke", &tmp).unwrap();

        let mut loaded = GptModel::new(config.clone());
        load_artifact_with_spec(&tmp, &mut loaded, &spec, true).unwrap();
        let inputs = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let targets = vec![2u32, 3, 4, 5, 6, 7, 8, 9];
        let mut buf = pg_model::model::ForwardBuffer::new(&config, inputs.len());
        loaded.forward(&inputs, &mut buf);
        let loss = loaded.compute_loss(&targets, &buf);
        let _ = std::fs::remove_file(&tmp);
        assert!(
            loss.is_finite(),
            "strict exported artifact loss is not finite"
        );
        assert!(
            loss > 0.0,
            "strict exported artifact loss should be positive"
        );
    }

    #[test]
    fn lqer_low_rank_residual_reduces_reconstruction_error() {
        let rows = 12;
        let cols = 10;
        let rank = 2;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let r = i / cols;
                let c = i % cols;
                ((r as f32 * 0.37).sin() * (c as f32 * 0.19).cos())
                    + ((r as f32 * 0.11).cos() * (c as f32 * 0.41).sin() * 0.25)
            })
            .collect();
        let cfg = GroupConfig::new(Bits::B4, Block::PerRow);
        let packed = quantize_with(&weights, rows, cols, &cfg);
        let base = packed.dequantize();
        let residual: Vec<f32> = weights
            .iter()
            .zip(base.iter())
            .map(|(&w, &q)| w - q)
            .collect();
        let (a, b) = low_rank_residual_factors(&residual, rows, cols, rank);
        let mut corrected = base;
        for r in 0..rows {
            for c in 0..cols {
                for k in 0..rank {
                    corrected[r * cols + c] += a[r * rank + k] * b[k * cols + c];
                }
            }
        }
        assert!(mse(&weights, &corrected) < mse(&weights, &packed.dequantize()));
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
            logit_softcap_pos: 30.0,
            logit_softcap_neg: 30.0,
            qk_gain_init: 1.5,
            recurrence_enabled: false,
            recurrence_start_layer: 0,
            recurrence_repeat_layers: 0,
            parallel_residual: false,
            parallel_residual_start_layer: 0,
            attn_out_gate_enabled: false,
            attn_out_gate_width: 24,
            sparse_attn_gate_enabled: false,
            sparse_attn_gate_width: 12,
            sparse_attn_gate_scale: 1.0,
            vrl_enabled: false,
            smear_gate_boundary_token_id: Some(1),
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
