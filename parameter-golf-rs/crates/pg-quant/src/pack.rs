use pg_core::error::{PgError, PgResult};

use crate::layout::{CompiledQuantLayoutManifest, QuantArchProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledQuantGroupKernel {
    pub name: &'static str,
    pub bits: u8,
    pub pack_kernel: &'static str,
    pub dequant_kernel: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledQuantKernelSet {
    pub arch_profile: QuantArchProfile,
    pub groups: Vec<CompiledQuantGroupKernel>,
}

impl CompiledQuantKernelSet {
    pub fn for_manifest(manifest: &CompiledQuantLayoutManifest) -> Self {
        let groups = manifest
            .groups
            .iter()
            .map(|group| CompiledQuantGroupKernel {
                name: group.name,
                bits: group.bits,
                pack_kernel: group.pack_kernel,
                dequant_kernel: group.dequant_kernel,
            })
            .collect::<Vec<_>>();
        let set = Self {
            arch_profile: manifest.arch_profile,
            groups,
        };
        set.validate_kernel_ids()
            .expect("compiled quant manifest should only contain supported kernel ids");
        set
    }

    pub fn from_arch_profile(arch_profile: QuantArchProfile) -> Self {
        Self {
            arch_profile,
            groups: Vec::new(),
        }
    }

    pub fn kernel_ids(&self) -> Vec<&'static str> {
        let mut ids = Vec::new();
        for group in &self.groups {
            if !ids.contains(&group.pack_kernel) {
                ids.push(group.pack_kernel);
            }
            if !ids.contains(&group.dequant_kernel) {
                ids.push(group.dequant_kernel);
            }
        }
        ids
    }

    pub fn validate_kernel_ids(&self) -> PgResult<()> {
        for group in &self.groups {
            validate_supported_bits(group.bits)?;
            let pack = expected_pack_kernel_id(group.bits, self.arch_profile);
            let dequant = expected_dequant_kernel_id(group.bits, self.arch_profile);
            if group.pack_kernel != pack {
                return Err(PgError::InvalidOp(format!(
                    "compiled quant group {} uses pack kernel {}, expected {}",
                    group.name, group.pack_kernel, pack
                )));
            }
            if group.dequant_kernel != dequant {
                return Err(PgError::InvalidOp(format!(
                    "compiled quant group {} uses dequant kernel {}, expected {}",
                    group.name, group.dequant_kernel, dequant
                )));
            }
        }
        Ok(())
    }

    pub fn pack_signed(&self, values: &[i8], bits: u8) -> PgResult<Vec<u8>> {
        pack_signed_by_bits(values, bits)
    }

    pub fn unpack_signed(&self, data: &[u8], count: usize, bits: u8) -> PgResult<Vec<i8>> {
        unpack_signed_by_bits(data, count, bits)
    }

    pub fn dequant_per_row(
        &self,
        data: &[u8],
        scale_data: &[u8],
        rows: usize,
        cols: usize,
        bits: u8,
        dest: &mut [f32],
    ) -> PgResult<()> {
        dequant_per_row_by_bits(data, scale_data, rows, cols, bits, dest)
    }
}

pub fn pack_signed_by_bits(values: &[i8], bits: u8) -> PgResult<Vec<u8>> {
    match bits {
        2 => Ok(pack_signed_per_row::<2>(values)),
        3 => Ok(pack_signed_per_row::<3>(values)),
        4 => Ok(pack_signed_per_row::<4>(values)),
        5 => Ok(pack_signed_per_row::<5>(values)),
        6 => Ok(pack_signed_per_row::<6>(values)),
        7 => Ok(pack_signed_per_row::<7>(values)),
        8 => Ok(pack_signed_per_row::<8>(values)),
        _ => Err(PgError::InvalidOp(format!(
            "unsupported pack bit width: {bits}; expected 2..=8"
        ))),
    }
}

pub fn unpack_signed_by_bits(data: &[u8], count: usize, bits: u8) -> PgResult<Vec<i8>> {
    match bits {
        2 => Ok(unpack_signed_per_row::<2>(data, count)),
        3 => Ok(unpack_signed_per_row::<3>(data, count)),
        4 => Ok(unpack_signed_per_row::<4>(data, count)),
        5 => Ok(unpack_signed_per_row::<5>(data, count)),
        6 => Ok(unpack_signed_per_row::<6>(data, count)),
        7 => Ok(unpack_signed_per_row::<7>(data, count)),
        8 => Ok(unpack_signed_per_row::<8>(data, count)),
        _ => Err(PgError::InvalidOp(format!(
            "unsupported unpack bit width: {bits}; expected 2..=8"
        ))),
    }
}

pub fn dequant_per_row_by_bits(
    data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    bits: u8,
    dest: &mut [f32],
) -> PgResult<()> {
    match bits {
        2 => dequant_per_row::<2>(data, scale_data, rows, cols, dest),
        3 => dequant_per_row::<3>(data, scale_data, rows, cols, dest),
        4 => dequant_per_row::<4>(data, scale_data, rows, cols, dest),
        5 => dequant_per_row::<5>(data, scale_data, rows, cols, dest),
        6 => dequant_per_row::<6>(data, scale_data, rows, cols, dest),
        7 => dequant_per_row::<7>(data, scale_data, rows, cols, dest),
        8 => dequant_per_row::<8>(data, scale_data, rows, cols, dest),
        _ => Err(PgError::InvalidOp(format!(
            "unsupported dequant bit width: {bits}; expected 2..=8"
        ))),
    }
}

pub fn pack_signed_per_row<const BITS: usize>(values: &[i8]) -> Vec<u8> {
    assert_supported_bits::<BITS>();
    let qmin = qmin_for_bits(BITS as u8);
    let mut out = vec![0u8; (values.len() * BITS + 7) / 8];
    let mut bit_pos = 0usize;
    for &value in values {
        let encoded = (value as i32 - qmin) as u32;
        for b in 0..BITS {
            if ((encoded >> b) & 1) != 0 {
                let dst = bit_pos + b;
                out[dst / 8] |= 1u8 << (dst % 8);
            }
        }
        bit_pos += BITS;
    }
    out
}

pub fn unpack_signed_per_row<const BITS: usize>(data: &[u8], count: usize) -> Vec<i8> {
    assert_supported_bits::<BITS>();
    let qmin = qmin_for_bits(BITS as u8);
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut encoded = 0u32;
        for b in 0..BITS {
            let src = bit_pos + b;
            if src / 8 < data.len() && (data[src / 8] & (1u8 << (src % 8))) != 0 {
                encoded |= 1u32 << b;
            }
        }
        out.push((encoded as i32 + qmin) as i8);
        bit_pos += BITS;
    }
    out
}

pub fn dequant_per_row<const BITS: usize>(
    data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    dest: &mut [f32],
) -> PgResult<()> {
    assert_supported_bits::<BITS>();
    if dest.len() != rows * cols {
        return Err(PgError::ShapeMismatch {
            expected: vec![rows, cols],
            got: vec![dest.len()],
        });
    }
    if scale_data.len() != rows * 2 {
        return Err(PgError::DataFormat(format!(
            "invalid dequant scale length: expected {}, got {}",
            rows * 2,
            scale_data.len()
        )));
    }
    let q = unpack_signed_per_row::<BITS>(data, rows * cols);
    for r in 0..rows {
        let scale_bits = u16::from_le_bytes([scale_data[r * 2], scale_data[r * 2 + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        for c in 0..cols {
            dest[r * cols + c] = q[r * cols + c] as f32 * scale;
        }
    }
    Ok(())
}

pub fn qmax_for_bits(bits: u8) -> i32 {
    assert!((2..=8).contains(&bits), "unsupported bit width: {bits}");
    (1i32 << (bits - 1)) - 1
}

pub fn qmin_for_bits(bits: u8) -> i32 {
    assert!((2..=8).contains(&bits), "unsupported bit width: {bits}");
    -(1i32 << (bits - 1))
}

fn assert_supported_bits<const BITS: usize>() {
    validate_supported_bits(BITS as u8).expect("unsupported compiled quant bit width");
}

fn validate_supported_bits(bits: u8) -> PgResult<()> {
    if (2..=8).contains(&bits) {
        Ok(())
    } else {
        Err(PgError::InvalidOp(format!(
            "unsupported compiled quant bit width: {bits}; expected 2..=8"
        )))
    }
}

fn expected_pack_kernel_id(bits: u8, arch_profile: QuantArchProfile) -> String {
    match arch_profile {
        QuantArchProfile::PortableCpu => format!("pack_signed_i{bits}_per_row_cpu"),
        QuantArchProfile::Sm90H100 => format!("pack_signed_i{bits}_per_row_sm90"),
    }
}

fn expected_dequant_kernel_id(bits: u8, arch_profile: QuantArchProfile) -> String {
    match arch_profile {
        QuantArchProfile::PortableCpu => format!("dequant_i{bits}_per_row_f16_scale_cpu"),
        QuantArchProfile::Sm90H100 => format!("dequant_i{bits}_per_row_f16_scale_sm90"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values_for_bits(bits: u8) -> Vec<i8> {
        let qmin = qmin_for_bits(bits);
        let qmax = qmax_for_bits(bits);
        (0..37)
            .map(|i| (qmin + (i * 5 % (qmax - qmin + 1))) as i8)
            .collect()
    }

    #[test]
    fn pack_unpack_roundtrips_supported_bits() {
        for bits in 2..=8 {
            let values = values_for_bits(bits);
            let packed = pack_signed_by_bits(&values, bits).unwrap();
            assert_eq!(packed.len(), (values.len() * bits as usize + 7) / 8);
            let unpacked = unpack_signed_by_bits(&packed, values.len(), bits).unwrap();
            assert_eq!(unpacked, values, "roundtrip failed for {bits} bits");
        }
    }

    #[test]
    fn dequant_per_row_uses_f16_row_scales() {
        let values = vec![-8, -1, 0, 7, -3, 2, 4, 5];
        let packed = pack_signed_per_row::<4>(&values);
        let mut scales = Vec::new();
        for scale in [0.5f32, 2.0] {
            scales.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        }
        let mut out = vec![0.0; values.len()];
        dequant_per_row::<4>(&packed, &scales, 2, 4, &mut out).unwrap();
        assert_eq!(out[0], -4.0);
        assert_eq!(out[3], 3.5);
        assert_eq!(out[4], -6.0);
        assert_eq!(out[7], 10.0);
    }

    #[test]
    fn kernel_set_carries_manifest_kernel_ids() {
        let quant_spec = pg_model::QuantSpec {
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            attn_gate_bits: 8,
            gptq_calibration_batches: 32,
            lqer: pg_model::spec::LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let config = pg_model::ModelConfig::default();
        let manifest =
            crate::layout::compile_quant_layout_manifest(&quant_spec, Some(&config)).unwrap();
        let kernels = CompiledQuantKernelSet::for_manifest(&manifest);
        kernels.validate_kernel_ids().unwrap();
        let ids = kernels.kernel_ids();
        assert!(ids.contains(&"pack_signed_i4_per_row_sm90"));
        assert!(ids.contains(&"dequant_i4_per_row_f16_scale_sm90"));
    }

    #[test]
    fn cpu_pack_dequant_microbench_smoke() {
        let rows = 64;
        let cols = 128;
        let values: Vec<i8> = (0..rows * cols).map(|i| ((i * 7) % 16) as i8 - 8).collect();
        let mut scales = Vec::with_capacity(rows * 2);
        for r in 0..rows {
            scales.extend_from_slice(
                &half::f16::from_f32(0.25 + r as f32 * 0.001)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
        let mut checksum = 0.0f32;
        let start = std::time::Instant::now();
        for _ in 0..16 {
            let packed = pack_signed_per_row::<4>(&values);
            let mut out = vec![0.0; rows * cols];
            dequant_per_row::<4>(&packed, &scales, rows, cols, &mut out).unwrap();
            checksum += out.iter().step_by(257).sum::<f32>();
        }
        let elapsed = start.elapsed();
        eprintln!(
            "cpu_pack_dequant_microbench_smoke: {:?} for {} values, checksum={:.6}",
            elapsed,
            values.len() * 16,
            checksum
        );
        assert!(elapsed.as_nanos() > 0);
        assert!(checksum.is_finite());
    }
}
