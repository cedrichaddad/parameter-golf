use pg_core::error::{PgError, PgResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheGolfLayout {
    Contiguous,
    Paged,
}

#[derive(Clone, Debug)]
pub struct CacheGolfKvCache {
    pub tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub k_bits: u8,
    pub v_bits: u8,
    pub block_size_tokens: usize,
    pub layout: CacheGolfLayout,
    pub k_packed: Vec<u8>,
    pub v_packed: Vec<u8>,
    pub k_scales: Vec<f32>,
    pub v_scales: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheGolfKvErrorStats {
    pub max_key_l2_error: f32,
    pub max_value_l2_error: f32,
    pub max_value_l2_norm: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn cachegolf_quantize_kv(
    k: &[f32],
    v: &[f32],
    tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
    k_bits: u8,
    v_bits: u8,
    block_size_tokens: usize,
    layout: CacheGolfLayout,
) -> PgResult<CacheGolfKvCache> {
    validate_bits(k_bits, "k_bits")?;
    validate_bits(v_bits, "v_bits")?;
    let elems = tokens
        .checked_mul(num_kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| PgError::InvalidOp("KV cache shape overflow".into()))?;
    if k.len() != elems {
        return Err(PgError::ShapeMismatch {
            expected: vec![tokens, num_kv_heads, head_dim],
            got: vec![k.len()],
        });
    }
    if v.len() != elems {
        return Err(PgError::ShapeMismatch {
            expected: vec![tokens, num_kv_heads, head_dim],
            got: vec![v.len()],
        });
    }
    let block_size_tokens = block_size_tokens.max(1);
    let stored_tokens = stored_tokens_for_layout(tokens, block_size_tokens, layout)?;
    let stored_elems = stored_tokens
        .checked_mul(num_kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| PgError::InvalidOp("KV cache stored shape overflow".into()))?;
    let mut k_scales = vec![1.0f32; num_kv_heads * head_dim];
    let mut v_scales = vec![1.0f32; stored_tokens * num_kv_heads];

    let k_qmax = signed_qmax(k_bits);
    for kv_head in 0..num_kv_heads {
        for dim in 0..head_dim {
            let mut max_abs = 0.0f32;
            for token in 0..tokens {
                max_abs =
                    max_abs.max(k[kv_offset(token, kv_head, dim, num_kv_heads, head_dim)].abs());
            }
            k_scales[k_channel_offset(kv_head, dim, head_dim)] = scale_for_max_abs(max_abs, k_qmax);
        }
    }

    let v_qmax = signed_qmax(v_bits);
    for token in 0..tokens {
        for kv_head in 0..num_kv_heads {
            let mut max_abs = 0.0f32;
            for dim in 0..head_dim {
                max_abs =
                    max_abs.max(v[kv_offset(token, kv_head, dim, num_kv_heads, head_dim)].abs());
            }
            v_scales[v_token_offset(token, kv_head, num_kv_heads)] =
                scale_for_max_abs(max_abs, v_qmax);
        }
    }

    let mut k_codes = vec![signed_qmax(k_bits) as u8; stored_elems];
    let mut v_codes = vec![signed_qmax(v_bits) as u8; stored_elems];
    for token in 0..tokens {
        for kv_head in 0..num_kv_heads {
            for dim in 0..head_dim {
                let idx = kv_offset(token, kv_head, dim, num_kv_heads, head_dim);
                let physical_idx = code_offset(
                    token,
                    kv_head,
                    dim,
                    num_kv_heads,
                    head_dim,
                    block_size_tokens,
                    layout,
                );
                let k_scale = k_scales[k_channel_offset(kv_head, dim, head_dim)];
                let v_scale = v_scales[v_token_offset(token, kv_head, num_kv_heads)];
                k_codes[physical_idx] = quantize_symmetric(k[idx], k_scale, k_bits);
                v_codes[physical_idx] = quantize_symmetric(v[idx], v_scale, v_bits);
            }
        }
    }

    Ok(CacheGolfKvCache {
        tokens,
        num_kv_heads,
        head_dim,
        k_bits,
        v_bits,
        block_size_tokens,
        layout,
        k_packed: pack_low_bits(&k_codes, k_bits),
        v_packed: pack_low_bits(&v_codes, v_bits),
        k_scales,
        v_scales,
    })
}

impl CacheGolfKvCache {
    pub fn stored_tokens(&self) -> usize {
        stored_tokens_for_layout(self.tokens, self.block_size_tokens, self.layout)
            .expect("CacheGolfKvCache stores a valid layout and block size")
    }

    pub fn page_table_bytes(&self) -> usize {
        match self.layout {
            CacheGolfLayout::Contiguous => 0,
            CacheGolfLayout::Paged => self
                .tokens
                .div_ceil(self.block_size_tokens.max(1))
                .saturating_mul(16),
        }
    }

    pub fn f16_scale_artifact_bytes(&self) -> usize {
        self.k_packed.len() + self.v_packed.len() + (self.k_scales.len() + self.v_scales.len()) * 2
    }

    pub fn f16_scale_runtime_bytes(&self) -> usize {
        self.f16_scale_artifact_bytes()
            .saturating_add(self.page_table_bytes())
    }

    pub fn try_k_value(&self, token: usize, kv_head: usize, dim: usize) -> PgResult<f32> {
        self.validate_value_index(token, kv_head, dim)?;
        validate_bits(self.k_bits, "k_bits")?;
        let idx = code_offset(
            token,
            kv_head,
            dim,
            self.num_kv_heads,
            self.head_dim,
            self.block_size_tokens,
            self.layout,
        );
        let code = unpack_low_bit_checked(&self.k_packed, idx, self.k_bits, "k_packed")?;
        let scale = *self
            .k_scales
            .get(k_channel_offset(kv_head, dim, self.head_dim))
            .ok_or_else(|| {
                PgError::InvalidOp(format!(
                    "k_scales missing entry for kv_head={kv_head}, dim={dim}"
                ))
            })?;
        Ok(dequantize_symmetric(code, scale, self.k_bits))
    }

    pub fn try_v_value(&self, token: usize, kv_head: usize, dim: usize) -> PgResult<f32> {
        self.validate_value_index(token, kv_head, dim)?;
        validate_bits(self.v_bits, "v_bits")?;
        let idx = code_offset(
            token,
            kv_head,
            dim,
            self.num_kv_heads,
            self.head_dim,
            self.block_size_tokens,
            self.layout,
        );
        let code = unpack_low_bit_checked(&self.v_packed, idx, self.v_bits, "v_packed")?;
        let scale = *self
            .v_scales
            .get(v_token_offset(token, kv_head, self.num_kv_heads))
            .ok_or_else(|| {
                PgError::InvalidOp(format!(
                    "v_scales missing entry for token={token}, kv_head={kv_head}"
                ))
            })?;
        Ok(dequantize_symmetric(code, scale, self.v_bits))
    }

    fn validate_value_index(&self, token: usize, kv_head: usize, dim: usize) -> PgResult<()> {
        if token >= self.tokens {
            return Err(PgError::ShapeMismatch {
                expected: vec![self.tokens, self.num_kv_heads, self.head_dim],
                got: vec![token + 1, self.num_kv_heads, self.head_dim],
            });
        }
        if kv_head >= self.num_kv_heads {
            return Err(PgError::ShapeMismatch {
                expected: vec![self.tokens, self.num_kv_heads, self.head_dim],
                got: vec![self.tokens, kv_head + 1, self.head_dim],
            });
        }
        if dim >= self.head_dim {
            return Err(PgError::ShapeMismatch {
                expected: vec![self.tokens, self.num_kv_heads, self.head_dim],
                got: vec![self.tokens, self.num_kv_heads, dim + 1],
            });
        }
        Ok(())
    }

    pub fn dequantize_kv(&self, k: &mut [f32], v: &mut [f32]) -> PgResult<()> {
        let elems = self.tokens * self.num_kv_heads * self.head_dim;
        if k.len() != elems {
            return Err(PgError::ShapeMismatch {
                expected: vec![self.tokens, self.num_kv_heads, self.head_dim],
                got: vec![k.len()],
            });
        }
        if v.len() != elems {
            return Err(PgError::ShapeMismatch {
                expected: vec![self.tokens, self.num_kv_heads, self.head_dim],
                got: vec![v.len()],
            });
        }
        for token in 0..self.tokens {
            for kv_head in 0..self.num_kv_heads {
                for dim in 0..self.head_dim {
                    let idx = kv_offset(token, kv_head, dim, self.num_kv_heads, self.head_dim);
                    k[idx] = self.try_k_value(token, kv_head, dim)?;
                    v[idx] = self.try_v_value(token, kv_head, dim)?;
                }
            }
        }
        Ok(())
    }
}

pub fn cachegolf_causal_attention_forward(
    q: &[f32],
    cache: &CacheGolfKvCache,
    output: &mut [f32],
    num_heads: usize,
) -> PgResult<()> {
    if cache.num_kv_heads == 0 || num_heads == 0 || num_heads % cache.num_kv_heads != 0 {
        return Err(PgError::InvalidOp(format!(
            "num_heads ({num_heads}) must be a nonzero multiple of num_kv_heads ({})",
            cache.num_kv_heads
        )));
    }
    let expected_q = cache.tokens * num_heads * cache.head_dim;
    if q.len() != expected_q {
        return Err(PgError::ShapeMismatch {
            expected: vec![cache.tokens, num_heads, cache.head_dim],
            got: vec![q.len()],
        });
    }
    if output.len() != expected_q {
        return Err(PgError::ShapeMismatch {
            expected: vec![cache.tokens, num_heads, cache.head_dim],
            got: vec![output.len()],
        });
    }

    let group = num_heads / cache.num_kv_heads;
    let scale = 1.0 / (cache.head_dim as f32).sqrt();
    for head in 0..num_heads {
        let kv_head = head / group;
        for token in 0..cache.tokens {
            let q_offset = (token * num_heads + head) * cache.head_dim;
            let out_offset = q_offset;
            let mut max_score = f32::NEG_INFINITY;
            for src in 0..=token {
                let mut dot = 0.0f32;
                for dim in 0..cache.head_dim {
                    dot += q[q_offset + dim] * cache.try_k_value(src, kv_head, dim)?;
                }
                max_score = max_score.max(dot * scale);
            }

            for dim in 0..cache.head_dim {
                output[out_offset + dim] = 0.0;
            }
            let mut sum_exp = 0.0f32;
            for src in 0..=token {
                let mut dot = 0.0f32;
                for dim in 0..cache.head_dim {
                    dot += q[q_offset + dim] * cache.try_k_value(src, kv_head, dim)?;
                }
                let weight = (dot * scale - max_score).exp();
                sum_exp += weight;
                for dim in 0..cache.head_dim {
                    output[out_offset + dim] += weight * cache.try_v_value(src, kv_head, dim)?;
                }
            }
            let inv_sum = 1.0 / sum_exp;
            for dim in 0..cache.head_dim {
                output[out_offset + dim] *= inv_sum;
            }
        }
    }
    Ok(())
}

pub fn cachegolf_kv_error_stats(
    cache: &CacheGolfKvCache,
    k: &[f32],
    v: &[f32],
) -> PgResult<CacheGolfKvErrorStats> {
    let elems = cache.tokens * cache.num_kv_heads * cache.head_dim;
    if k.len() != elems {
        return Err(PgError::ShapeMismatch {
            expected: vec![cache.tokens, cache.num_kv_heads, cache.head_dim],
            got: vec![k.len()],
        });
    }
    if v.len() != elems {
        return Err(PgError::ShapeMismatch {
            expected: vec![cache.tokens, cache.num_kv_heads, cache.head_dim],
            got: vec![v.len()],
        });
    }

    let mut max_key_l2_error = 0.0f32;
    let mut max_value_l2_error = 0.0f32;
    let mut max_value_l2_norm = 0.0f32;
    for token in 0..cache.tokens {
        for kv_head in 0..cache.num_kv_heads {
            let mut key_err2 = 0.0f32;
            let mut value_err2 = 0.0f32;
            let mut value_norm2 = 0.0f32;
            for dim in 0..cache.head_dim {
                let idx = kv_offset(token, kv_head, dim, cache.num_kv_heads, cache.head_dim);
                let dk = cache.try_k_value(token, kv_head, dim)? - k[idx];
                let dv = cache.try_v_value(token, kv_head, dim)? - v[idx];
                key_err2 += dk * dk;
                value_err2 += dv * dv;
                value_norm2 += v[idx] * v[idx];
            }
            max_key_l2_error = max_key_l2_error.max(key_err2.sqrt());
            max_value_l2_error = max_value_l2_error.max(value_err2.sqrt());
            max_value_l2_norm = max_value_l2_norm.max(value_norm2.sqrt());
        }
    }
    Ok(CacheGolfKvErrorStats {
        max_key_l2_error,
        max_value_l2_error,
        max_value_l2_norm,
    })
}

pub fn cachegolf_attention_error_bound(
    q_l2_norm: f32,
    stats: CacheGolfKvErrorStats,
    head_dim: usize,
) -> f32 {
    stats.max_value_l2_error
        + 2.0 * q_l2_norm * stats.max_key_l2_error / (head_dim as f32).sqrt()
            * stats.max_value_l2_norm
}

fn validate_bits(bits: u8, name: &str) -> PgResult<()> {
    if (2..=8).contains(&bits) {
        Ok(())
    } else {
        Err(PgError::InvalidOp(format!(
            "{name} must be in 2..=8, got {bits}"
        )))
    }
}

fn stored_tokens_for_layout(
    tokens: usize,
    block_size_tokens: usize,
    layout: CacheGolfLayout,
) -> PgResult<usize> {
    match layout {
        CacheGolfLayout::Contiguous => Ok(tokens),
        CacheGolfLayout::Paged => tokens
            .div_ceil(block_size_tokens.max(1))
            .checked_mul(block_size_tokens.max(1))
            .ok_or_else(|| PgError::InvalidOp("paged KV cache token capacity overflow".into())),
    }
}

fn signed_qmax(bits: u8) -> i32 {
    (1i32 << (bits - 1)) - 1
}

fn scale_for_max_abs(max_abs: f32, qmax: i32) -> f32 {
    if max_abs > 0.0 {
        max_abs / qmax as f32
    } else {
        1.0
    }
}

fn quantize_symmetric(value: f32, scale: f32, bits: u8) -> u8 {
    let qmax = signed_qmax(bits);
    let signed = (value / scale).round().clamp(-(qmax as f32), qmax as f32) as i32;
    (signed + qmax) as u8
}

fn dequantize_symmetric(code: u8, scale: f32, bits: u8) -> f32 {
    let signed = code as i32 - signed_qmax(bits);
    signed as f32 * scale
}

fn pack_low_bits(codes: &[u8], bits: u8) -> Vec<u8> {
    let mut packed = vec![0u8; (codes.len() * bits as usize).div_ceil(8)];
    let mask = (1u16 << bits) - 1;
    for (index, &code) in codes.iter().enumerate() {
        let value = (code as u16) & mask;
        let bit_offset = index * bits as usize;
        let byte_offset = bit_offset / 8;
        let shift = bit_offset % 8;
        packed[byte_offset] |= (value << shift) as u8;
        if shift + bits as usize > 8 {
            packed[byte_offset + 1] |= (value >> (8 - shift)) as u8;
        }
    }
    packed
}

fn unpack_low_bit(packed: &[u8], index: usize, bits: u8) -> u8 {
    let bit_offset = index * bits as usize;
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    let mut value = (packed[byte_offset] as u16) >> shift;
    if shift + bits as usize > 8 {
        value |= (packed[byte_offset + 1] as u16) << (8 - shift);
    }
    (value & ((1u16 << bits) - 1)) as u8
}

fn unpack_low_bit_checked(packed: &[u8], index: usize, bits: u8, label: &str) -> PgResult<u8> {
    let bit_offset = index
        .checked_mul(bits as usize)
        .ok_or_else(|| PgError::InvalidOp(format!("{label} bit offset overflow")))?;
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    if byte_offset >= packed.len() {
        return Err(PgError::InvalidOp(format!(
            "{label} too short for low-bit index {index}: byte_offset={byte_offset}, len={}",
            packed.len()
        )));
    }
    let crosses_byte = shift + bits as usize > 8;
    if crosses_byte && byte_offset + 1 >= packed.len() {
        return Err(PgError::InvalidOp(format!(
            "{label} too short for straddled low-bit index {index}: byte_offset={}, len={}",
            byte_offset + 1,
            packed.len()
        )));
    }
    Ok(unpack_low_bit(packed, index, bits))
}

fn kv_offset(
    token: usize,
    kv_head: usize,
    dim: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> usize {
    (token * num_kv_heads + kv_head) * head_dim + dim
}

fn code_offset(
    token: usize,
    kv_head: usize,
    dim: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size_tokens: usize,
    layout: CacheGolfLayout,
) -> usize {
    match layout {
        CacheGolfLayout::Contiguous => kv_offset(token, kv_head, dim, num_kv_heads, head_dim),
        CacheGolfLayout::Paged => {
            let block_size_tokens = block_size_tokens.max(1);
            let block = token / block_size_tokens;
            let token_in_block = token % block_size_tokens;
            let block_stride = block_size_tokens * num_kv_heads * head_dim;
            block * block_stride + (kv_head * head_dim + dim) * block_size_tokens + token_in_block
        }
    }
}

fn k_channel_offset(kv_head: usize, dim: usize, head_dim: usize) -> usize {
    kv_head * head_dim + dim
}

fn v_token_offset(token: usize, kv_head: usize, num_kv_heads: usize) -> usize {
    token * num_kv_heads + kv_head
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::causal_attention_forward;

    #[test]
    fn cachegolf_quantizes_keys_per_channel_and_values_per_token() {
        let tokens = 3;
        let kv_heads = 2;
        let head_dim = 4;
        let k = deterministic_values(tokens * kv_heads * head_dim, 0.17);
        let v = deterministic_values(tokens * kv_heads * head_dim, 0.31);
        let cache = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            4,
            3,
            2,
            CacheGolfLayout::Paged,
        )
        .unwrap();

        assert_eq!(cache.k_scales.len(), kv_heads * head_dim);
        assert_eq!(cache.stored_tokens(), 4);
        assert_eq!(cache.v_scales.len(), cache.stored_tokens() * kv_heads);
        assert_eq!(
            cache.k_packed.len(),
            (cache.stored_tokens() * kv_heads * head_dim * 4usize).div_ceil(8)
        );
        assert_eq!(
            cache.v_packed.len(),
            (cache.stored_tokens() * kv_heads * head_dim * 3usize).div_ceil(8)
        );
        assert_eq!(
            cache.f16_scale_artifact_bytes(),
            cache.k_packed.len()
                + cache.v_packed.len()
                + (kv_heads * head_dim + cache.stored_tokens() * kv_heads) * 2
        );
        assert_eq!(cache.page_table_bytes(), 32);
        assert_eq!(
            cache.f16_scale_runtime_bytes(),
            cache.f16_scale_artifact_bytes() + cache.page_table_bytes()
        );
    }

    #[test]
    fn cachegolf_dequant_attention_matches_dequantized_reference() {
        let tokens = 5;
        let heads = 4;
        let kv_heads = 2;
        let head_dim = 6;
        let q = deterministic_values(tokens * heads * head_dim, 0.13);
        let k = deterministic_values(tokens * kv_heads * head_dim, 0.23);
        let v = deterministic_values(tokens * kv_heads * head_dim, 0.37);
        let cache = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            5,
            4,
            4,
            CacheGolfLayout::Contiguous,
        )
        .unwrap();

        let mut deq_k = vec![0.0f32; k.len()];
        let mut deq_v = vec![0.0f32; v.len()];
        cache.dequantize_kv(&mut deq_k, &mut deq_v).unwrap();

        let mut expected = vec![0.0f32; q.len()];
        causal_attention_forward(
            &q,
            &deq_k,
            &deq_v,
            &mut expected,
            tokens,
            heads,
            kv_heads,
            head_dim,
        );
        let mut got = vec![0.0f32; q.len()];
        cachegolf_causal_attention_forward(&q, &cache, &mut got, heads).unwrap();

        assert_close_slice(&got, &expected, 2e-6, "dequant attention");
    }

    #[test]
    fn cachegolf_attention_error_bound_covers_observed_error() {
        let tokens = 6;
        let heads = 4;
        let kv_heads = 2;
        let head_dim = 8;
        let q = deterministic_values(tokens * heads * head_dim, 0.19);
        let k = deterministic_values(tokens * kv_heads * head_dim, 0.29);
        let v = deterministic_values(tokens * kv_heads * head_dim, 0.43);
        let cache = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            4,
            3,
            3,
            CacheGolfLayout::Paged,
        )
        .unwrap();

        let mut full = vec![0.0f32; q.len()];
        causal_attention_forward(&q, &k, &v, &mut full, tokens, heads, kv_heads, head_dim);
        let mut quantized = vec![0.0f32; q.len()];
        cachegolf_causal_attention_forward(&q, &cache, &mut quantized, heads).unwrap();
        let stats = cachegolf_kv_error_stats(&cache, &k, &v).unwrap();

        for token in 0..tokens {
            for head in 0..heads {
                let offset = (token * heads + head) * head_dim;
                let q_norm = l2_norm(&q[offset..offset + head_dim]);
                let bound = cachegolf_attention_error_bound(q_norm, stats, head_dim);
                let observed = l2_distance(
                    &full[offset..offset + head_dim],
                    &quantized[offset..offset + head_dim],
                );
                assert!(
                    observed <= bound + 1e-5,
                    "observed output error {observed} exceeded bound {bound} at token {token} head {head}"
                );
            }
        }
    }

    #[test]
    fn cachegolf_more_bits_reduce_reconstruction_error() {
        let tokens = 8;
        let kv_heads = 2;
        let head_dim = 5;
        let k = deterministic_values(tokens * kv_heads * head_dim, 0.11);
        let v = deterministic_values(tokens * kv_heads * head_dim, 0.47);
        let low = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            2,
            2,
            4,
            CacheGolfLayout::Contiguous,
        )
        .unwrap();
        let high = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            6,
            6,
            4,
            CacheGolfLayout::Contiguous,
        )
        .unwrap();

        let low_stats = cachegolf_kv_error_stats(&low, &k, &v).unwrap();
        let high_stats = cachegolf_kv_error_stats(&high, &k, &v).unwrap();
        assert!(high_stats.max_key_l2_error < low_stats.max_key_l2_error);
        assert!(high_stats.max_value_l2_error < low_stats.max_value_l2_error);
    }

    #[test]
    fn cachegolf_paged_layout_uses_blocked_physical_order() {
        let tokens = 5;
        let kv_heads = 2;
        let head_dim = 3;
        let k = deterministic_values(tokens * kv_heads * head_dim, 0.21);
        let v = deterministic_values(tokens * kv_heads * head_dim, 0.39);
        let contiguous = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            4,
            4,
            2,
            CacheGolfLayout::Contiguous,
        )
        .unwrap();
        let paged = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            4,
            4,
            2,
            CacheGolfLayout::Paged,
        )
        .unwrap();

        assert_eq!(contiguous.stored_tokens(), tokens);
        assert_eq!(paged.stored_tokens(), 6);
        assert_ne!(paged.k_packed, contiguous.k_packed);
        assert_ne!(paged.v_packed, contiguous.v_packed);

        let mut contiguous_k = vec![0.0f32; k.len()];
        let mut contiguous_v = vec![0.0f32; v.len()];
        let mut paged_k = vec![0.0f32; k.len()];
        let mut paged_v = vec![0.0f32; v.len()];
        contiguous
            .dequantize_kv(&mut contiguous_k, &mut contiguous_v)
            .unwrap();
        paged.dequantize_kv(&mut paged_k, &mut paged_v).unwrap();

        assert_close_slice(&paged_k, &contiguous_k, 0.0, "paged k dequant");
        assert_close_slice(&paged_v, &contiguous_v, 0.0, "paged v dequant");
        assert_eq!(paged.page_table_bytes(), 48);
    }

    #[test]
    fn cachegolf_accessors_return_errors_for_bad_indices_and_malformed_buffers() {
        let tokens = 3;
        let kv_heads = 2;
        let head_dim = 4;
        let k = deterministic_values(tokens * kv_heads * head_dim, 0.21);
        let v = deterministic_values(tokens * kv_heads * head_dim, 0.39);
        let mut cache = cachegolf_quantize_kv(
            &k,
            &v,
            tokens,
            kv_heads,
            head_dim,
            4,
            4,
            2,
            CacheGolfLayout::Paged,
        )
        .unwrap();

        assert!(cache.try_k_value(tokens, 0, 0).is_err());
        assert!(cache.try_k_value(0, kv_heads, 0).is_err());
        assert!(cache.try_v_value(0, 0, head_dim).is_err());

        cache.k_packed.clear();
        assert!(cache.try_k_value(0, 0, 0).is_err());
        assert!(cachegolf_kv_error_stats(&cache, &k, &v).is_err());
    }

    fn deterministic_values(n: usize, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f32 + 1.0;
                0.8 * (x * phase).sin() + 0.25 * (x * phase * 0.41).cos()
            })
            .collect()
    }

    fn l2_norm(values: &[f32]) -> f32 {
        values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    fn assert_close_slice(got: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let allowed = tol * e.abs().max(g.abs()).max(1.0);
            assert!(
                (g - e).abs() <= allowed,
                "{label} mismatch at {i}: got {g}, expected {e}, diff {} allowed {allowed}",
                (g - e).abs()
            );
        }
    }
}
