use pg_core::error::{PgError, PgResult};
use pg_model::{ModelConfig, QuantSpec};
use pg_quant_macros::quant_layout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledQuantLayout {
    pub name: &'static str,
    pub matrix_bits: u8,
    pub mlp_bits: u8,
    pub embed_bits: u8,
    pub attn_gate_bits: u8,
    pub gptq_calibration_batches: usize,
    pub target_artifact_bytes: usize,
    pub lqer_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledQuantGroupManifest {
    pub name: &'static str,
    pub bits: u8,
    pub block: &'static str,
    pub rows: Option<usize>,
    pub cols: Option<usize>,
    pub packed_weight_bytes: Option<usize>,
    pub scale_bytes: Option<usize>,
    pub lqer_bytes: Option<usize>,
    pub pack_kernel: &'static str,
    pub dequant_kernel: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledQuantLayoutManifest {
    pub layout: CompiledQuantLayout,
    pub generated_from: &'static str,
    pub pack_format: &'static str,
    pub dequant_format: &'static str,
    pub groups: Vec<CompiledQuantGroupManifest>,
    pub estimated_raw_weight_bytes: Option<usize>,
    pub target_artifact_bytes: usize,
    pub fingerprint_crc32: String,
}

impl CompiledQuantLayout {
    pub fn validate(self) -> PgResult<Self> {
        for (name, bits) in [
            ("matrix_bits", self.matrix_bits),
            ("mlp_bits", self.mlp_bits),
            ("embed_bits", self.embed_bits),
            ("attn_gate_bits", self.attn_gate_bits),
        ] {
            if !(2..=8).contains(&bits) {
                return Err(PgError::InvalidOp(format!(
                    "compiled quant layout {} has unsupported {name}={bits}; expected 2..=8",
                    self.name
                )));
            }
        }
        Ok(self)
    }

    pub const fn uses_mixed_mlp_bits(self) -> bool {
        self.matrix_bits != self.mlp_bits
    }

    pub fn matches_quant_spec(self, quant_spec: &QuantSpec) -> bool {
        quant_spec.matrix_bits == self.matrix_bits
            && quant_spec.mlp_bits == self.mlp_bits
            && quant_spec.embed_bits == self.embed_bits
            && quant_spec.attn_gate_bits == self.attn_gate_bits
            && quant_spec.gptq_calibration_batches == self.gptq_calibration_batches
            && quant_spec.target_artifact_bytes == self.target_artifact_bytes
            && quant_spec.lqer.enabled == self.lqer_enabled
    }

    pub fn validate_quant_spec(self, quant_spec: &QuantSpec) -> PgResult<Self> {
        let layout = self.validate()?;
        if !layout.matches_quant_spec(quant_spec) {
            return Err(PgError::InvalidOp(format!(
                "compiled quant layout {} does not match QuantSpec: layout bits={}/{}/{} gate={} gptq_batches={} target_bytes={} lqer={}, spec bits={}/{}/{} gate={} gptq_batches={} target_bytes={} lqer={}",
                layout.name,
                layout.matrix_bits,
                layout.mlp_bits,
                layout.embed_bits,
                layout.attn_gate_bits,
                layout.gptq_calibration_batches,
                layout.target_artifact_bytes,
                layout.lqer_enabled,
                quant_spec.matrix_bits,
                quant_spec.mlp_bits,
                quant_spec.embed_bits,
                quant_spec.attn_gate_bits,
                quant_spec.gptq_calibration_batches,
                quant_spec.target_artifact_bytes,
                quant_spec.lqer.enabled
            )));
        }
        Ok(layout)
    }
}

impl CompiledQuantLayoutManifest {
    pub fn metadata_json(&self) -> String {
        let groups = self
            .groups
            .iter()
            .map(|group| {
                format!(
                    "{{\"name\":\"{}\",\"bits\":{},\"block\":\"{}\",\"rows\":{},\"cols\":{},\"packed_weight_bytes\":{},\"scale_bytes\":{},\"lqer_bytes\":{},\"pack_kernel\":\"{}\",\"dequant_kernel\":\"{}\"}}",
                    group.name,
                    group.bits,
                    group.block,
                    json_usize_or_null(group.rows),
                    json_usize_or_null(group.cols),
                    json_usize_or_null(group.packed_weight_bytes),
                    json_usize_or_null(group.scale_bytes),
                    json_usize_or_null(group.lqer_bytes),
                    group.pack_kernel,
                    group.dequant_kernel
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"layout\":\"{}\",\"generated_from\":\"{}\",\"pack_format\":\"{}\",\"dequant_format\":\"{}\",\"target_artifact_bytes\":{},\"estimated_raw_weight_bytes\":{},\"fingerprint_crc32\":\"{}\",\"groups\":[{}]}}",
            self.layout.name,
            self.generated_from,
            self.pack_format,
            self.dequant_format,
            self.target_artifact_bytes,
            json_usize_or_null(self.estimated_raw_weight_bytes),
            self.fingerprint_crc32,
            groups
        )
    }
}

pub const FRONTIER_1855_MIXED_INT5_INT6_LAYOUT: CompiledQuantLayout = quant_layout! {
    name: "frontier_1855_mixed_int5_int6",
    matrix_bits: 6,
    mlp_bits: 5,
    embed_bits: 7,
    attn_gate_bits: 8,
    gptq_calibration_batches: 16,
    target_artifact_bytes: 16_000_000,
    lqer_enabled: true,
};

pub const FRONTIER_2135_MIXED_INT5_INT4_LAYOUT: CompiledQuantLayout = quant_layout! {
    name: "frontier_2135_mixed_int5_int4",
    matrix_bits: 5,
    mlp_bits: 4,
    embed_bits: 6,
    attn_gate_bits: 8,
    gptq_calibration_batches: 32,
    target_artifact_bytes: 16_000_000,
    lqer_enabled: true,
};

pub const FRONTIER_2135_CALIB32_INT6_INT7_LAYOUT: CompiledQuantLayout = quant_layout! {
    name: "frontier_2135_calib32_int6_int7",
    matrix_bits: 6,
    mlp_bits: 6,
    embed_bits: 7,
    attn_gate_bits: 8,
    gptq_calibration_batches: 32,
    target_artifact_bytes: 16_000_000,
    lqer_enabled: true,
};

pub fn compiled_layout_for_quant_spec(quant_spec: &QuantSpec) -> Option<CompiledQuantLayout> {
    [
        FRONTIER_1855_MIXED_INT5_INT6_LAYOUT,
        FRONTIER_2135_MIXED_INT5_INT4_LAYOUT,
        FRONTIER_2135_CALIB32_INT6_INT7_LAYOUT,
    ]
    .into_iter()
    .find(|layout| layout.matches_quant_spec(quant_spec))
}

pub fn compile_quant_layout_manifest(
    quant_spec: &QuantSpec,
    model_config: Option<&ModelConfig>,
) -> PgResult<CompiledQuantLayoutManifest> {
    let layout = compiled_layout_for_quant_spec(quant_spec).unwrap_or(CompiledQuantLayout {
        name: "spec_generated_quant_layout",
        matrix_bits: quant_spec.matrix_bits,
        mlp_bits: quant_spec.mlp_bits,
        embed_bits: quant_spec.embed_bits,
        attn_gate_bits: quant_spec.attn_gate_bits,
        gptq_calibration_batches: quant_spec.gptq_calibration_batches,
        target_artifact_bytes: quant_spec.target_artifact_bytes,
        lqer_enabled: quant_spec.lqer.enabled,
    });
    validate_export_bits("matrix_bits", layout.matrix_bits)?;
    validate_export_bits("mlp_bits", layout.mlp_bits)?;
    validate_export_bits("embed_bits", layout.embed_bits)?;
    validate_export_bits("attn_gate_bits", layout.attn_gate_bits)?;
    validate_lqer_bits("lqer.a_bits", quant_spec.lqer.a_bits)?;
    validate_lqer_bits("lqer.b_bits", quant_spec.lqer.b_bits)?;
    layout.validate_quant_spec(quant_spec)?;

    let mut groups = Vec::new();
    if let Some(config) = model_config {
        let n = config.num_layers;
        let d = config.model_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;
        groups.push(group_manifest(
            "qo_bank.q",
            layout.matrix_bits,
            Some((n * d, d)),
            quant_spec,
        ));
        groups.push(group_manifest(
            "qo_bank.o",
            layout.matrix_bits,
            Some((n * d, d)),
            quant_spec,
        ));
        groups.push(group_manifest(
            "kv_bank.k",
            layout.matrix_bits,
            Some((n * kv, d)),
            quant_spec,
        ));
        groups.push(group_manifest(
            "kv_bank.v",
            layout.matrix_bits,
            Some((n * kv, d)),
            quant_spec,
        ));
        groups.push(group_manifest(
            "mlp_up_bank",
            layout.mlp_bits,
            Some((n * mlp, d)),
            quant_spec,
        ));
        groups.push(group_manifest(
            "mlp_down_bank",
            layout.mlp_bits,
            Some((n * d, mlp)),
            quant_spec,
        ));
        groups.push(group_manifest(
            "tok_emb",
            layout.embed_bits,
            Some((config.vocab_size, d)),
            quant_spec,
        ));
    } else {
        for (name, bits) in [
            ("qo_bank.q", layout.matrix_bits),
            ("qo_bank.o", layout.matrix_bits),
            ("kv_bank.k", layout.matrix_bits),
            ("kv_bank.v", layout.matrix_bits),
            ("mlp_up_bank", layout.mlp_bits),
            ("mlp_down_bank", layout.mlp_bits),
            ("tok_emb", layout.embed_bits),
        ] {
            groups.push(group_manifest(name, bits, None, quant_spec));
        }
    }

    let estimated_raw_weight_bytes = groups.iter().try_fold(0usize, |acc, group| {
        Some(acc + group.packed_weight_bytes? + group.scale_bytes? + group.lqer_bytes.unwrap_or(0))
    });
    let generated_from = if compiled_layout_for_quant_spec(quant_spec).is_some() {
        "quant_layout_proc_macro"
    } else {
        "runtime_quant_spec_compiler"
    };
    let mut manifest = CompiledQuantLayoutManifest {
        layout,
        generated_from,
        pack_format: "signed_packed_int_per_row_f16_scale",
        dequant_format: "per_row_scale_dequant_add_optional_lqer",
        groups,
        estimated_raw_weight_bytes,
        target_artifact_bytes: layout.target_artifact_bytes,
        fingerprint_crc32: String::new(),
    };
    manifest.fingerprint_crc32 = manifest_crc32(&manifest);
    Ok(manifest)
}

fn group_manifest(
    name: &'static str,
    bits: u8,
    shape: Option<(usize, usize)>,
    quant_spec: &QuantSpec,
) -> CompiledQuantGroupManifest {
    let (rows, cols) = shape
        .map(|(rows, cols)| (Some(rows), Some(cols)))
        .unwrap_or((None, None));
    let packed_weight_bytes = shape.map(|(rows, cols)| div_ceil(rows * cols * bits as usize, 8));
    let scale_bytes = rows.map(|rows| rows * 2);
    let lqer_bytes = shape.map(|(rows, cols)| lqer_raw_bytes(rows, cols, quant_spec));
    CompiledQuantGroupManifest {
        name,
        bits,
        block: "per_row",
        rows,
        cols,
        packed_weight_bytes,
        scale_bytes,
        lqer_bytes,
        pack_kernel: "pack_signed_values",
        dequant_kernel: "dequant_packed_group",
    }
}

fn lqer_raw_bytes(rows: usize, cols: usize, quant_spec: &QuantSpec) -> usize {
    if !quant_spec.lqer.enabled || quant_spec.lqer.rank == 0 {
        return 0;
    }
    let rank = quant_spec.lqer.rank;
    let a_weight = div_ceil(rows * rank * quant_spec.lqer.a_bits as usize, 8);
    let b_weight = div_ceil(rank * cols * quant_spec.lqer.b_bits as usize, 8);
    let a_scale = rows * 2;
    let b_scale = rank * 2;
    a_weight + b_weight + a_scale + b_scale
}

fn validate_export_bits(field: &str, bits: u8) -> PgResult<()> {
    if !(4..=8).contains(&bits) {
        return Err(PgError::InvalidOp(format!(
            "quant layout compiler does not have export pack/dequant support for {field}={bits}; expected 4..=8"
        )));
    }
    Ok(())
}

fn validate_lqer_bits(field: &str, bits: u8) -> PgResult<()> {
    if !(2..=8).contains(&bits) {
        return Err(PgError::InvalidOp(format!(
            "quant layout compiler does not have LQER pack/dequant support for {field}={bits}; expected 2..=8"
        )));
    }
    Ok(())
}

fn div_ceil(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}

fn json_usize_or_null(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn manifest_crc32(manifest: &CompiledQuantLayoutManifest) -> String {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(manifest.layout.name.as_bytes());
    hasher.update(manifest.generated_from.as_bytes());
    hasher.update(manifest.pack_format.as_bytes());
    hasher.update(manifest.dequant_format.as_bytes());
    hasher.update(&manifest.target_artifact_bytes.to_le_bytes());
    if let Some(bytes) = manifest.estimated_raw_weight_bytes {
        hasher.update(&bytes.to_le_bytes());
    }
    for group in &manifest.groups {
        hasher.update(group.name.as_bytes());
        hasher.update(&[group.bits]);
        for value in [
            group.rows,
            group.cols,
            group.packed_weight_bytes,
            group.scale_bytes,
            group.lqer_bytes,
        ] {
            match value {
                Some(value) => hasher.update(&value.to_le_bytes()),
                None => hasher.update(&0usize.to_le_bytes()),
            }
        }
    }
    format!("{:08x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_layouts_are_compiled_and_valid() {
        let legacy = FRONTIER_1855_MIXED_INT5_INT6_LAYOUT.validate().unwrap();
        assert_eq!(legacy.matrix_bits, 6);
        assert_eq!(legacy.mlp_bits, 5);
        assert_eq!(legacy.embed_bits, 7);
        assert_eq!(legacy.gptq_calibration_batches, 16);
        assert!(legacy.uses_mixed_mlp_bits());
        assert!(legacy.lqer_enabled);

        let current = FRONTIER_2135_MIXED_INT5_INT4_LAYOUT.validate().unwrap();
        assert_eq!(current.matrix_bits, 5);
        assert_eq!(current.mlp_bits, 4);
        assert_eq!(current.embed_bits, 6);
        assert_eq!(current.gptq_calibration_batches, 32);
        assert!(current.uses_mixed_mlp_bits());
        assert!(current.lqer_enabled);

        let pr2135 = FRONTIER_2135_CALIB32_INT6_INT7_LAYOUT.validate().unwrap();
        assert_eq!(pr2135.matrix_bits, 6);
        assert_eq!(pr2135.mlp_bits, 6);
        assert_eq!(pr2135.embed_bits, 7);
        assert_eq!(pr2135.gptq_calibration_batches, 32);
        assert!(!pr2135.uses_mixed_mlp_bits());
        assert!(pr2135.lqer_enabled);
    }

    #[test]
    fn compiled_layout_matches_current_record_quant_spec() {
        let quant_spec = QuantSpec {
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            attn_gate_bits: 8,
            gptq_calibration_batches: 32,
            lqer: pg_model::spec::LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let layout = compiled_layout_for_quant_spec(&quant_spec)
            .expect("2135 quant spec should have a compiled layout");
        assert_eq!(layout.name, "frontier_2135_mixed_int5_int4");
        layout.validate_quant_spec(&quant_spec).unwrap();
    }

    #[test]
    fn compiled_layout_matches_upstream_pr2135_quant_spec() {
        let quant_spec = QuantSpec {
            matrix_bits: 6,
            mlp_bits: 6,
            embed_bits: 7,
            attn_gate_bits: 8,
            gptq_calibration_batches: 32,
            lqer: pg_model::spec::LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let layout = compiled_layout_for_quant_spec(&quant_spec)
            .expect("upstream PR2135 quant spec should have a compiled layout");
        assert_eq!(layout.name, "frontier_2135_calib32_int6_int7");
        layout.validate_quant_spec(&quant_spec).unwrap();
    }

    #[test]
    fn quant_layout_compiler_emits_model_manifest_and_fingerprint() {
        let quant_spec = QuantSpec {
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            attn_gate_bits: 8,
            gptq_calibration_batches: 32,
            lqer: pg_model::spec::LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let config = pg_model::ModelConfig::default();
        let manifest = compile_quant_layout_manifest(&quant_spec, Some(&config)).unwrap();
        assert_eq!(manifest.layout.name, "frontier_2135_mixed_int5_int4");
        assert_eq!(manifest.groups.len(), 7);
        assert!(manifest.estimated_raw_weight_bytes.unwrap() > 0);
        assert_eq!(manifest.fingerprint_crc32.len(), 8);
        let metadata = manifest.metadata_json();
        assert!(metadata.contains("\"pack_format\":\"signed_packed_int_per_row_f16_scale\""));
        assert!(metadata.contains("\"name\":\"mlp_down_bank\""));
    }

    #[test]
    fn quant_layout_compiler_rejects_export_unsupported_main_bits() {
        let quant_spec = QuantSpec {
            matrix_bits: 3,
            mlp_bits: 4,
            embed_bits: 6,
            attn_gate_bits: 8,
            gptq_calibration_batches: 32,
            lqer: pg_model::spec::LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let err = compile_quant_layout_manifest(&quant_spec, None).unwrap_err();
        assert!(err.to_string().contains("matrix_bits=3"));
    }
}
