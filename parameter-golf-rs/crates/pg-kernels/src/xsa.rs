/// XSA (Exclusive Self Attention) — removes self-value redundancy.
///
/// Standard attention output y_i has high cosine similarity with the token's
/// own value vector v_i. XSA projects out this component:
///   z_i = y_i - (y_i^T v_i / ||v_i||²) * v_i
///
/// Applied to the last 4 layers (XSA4). Zero new parameters, negligible compute.
/// BPB gain: ~0.003–0.005.
///
/// GQA-aware: with 8 heads and 4 KV heads (group=2), we reshape to
/// [B, T, Hkv, group, D] to avoid repeat_interleave.

/// Forward: project out self-value component.
/// y: [batch*seq, num_heads, head_dim] — attention output
/// v: [batch*seq, num_kv_heads, head_dim] — value vectors (before GQA expansion)
/// output: same shape as y
pub fn xsa_forward(
    y: &[f32],
    v: &[f32],
    output: &mut [f32],
    tokens: usize, // batch * seq_len
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let group = num_heads / num_kv_heads;

    for t in 0..tokens {
        for kv_h in 0..num_kv_heads {
            // Get v vector for this KV head
            let v_offset = (t * num_kv_heads + kv_h) * head_dim;
            let v_slice = &v[v_offset..v_offset + head_dim];

            // ||v||²
            let v_norm_sq: f32 = v_slice.iter().map(|&x| x * x).sum::<f32>() + 1e-8;

            // For each Q head in this group
            for g in 0..group {
                let h = kv_h * group + g;
                let y_offset = (t * num_heads + h) * head_dim;
                let y_slice = &y[y_offset..y_offset + head_dim];

                // y^T v
                let dot: f32 = y_slice
                    .iter()
                    .zip(v_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let coeff = dot / v_norm_sq;

                // z = y - coeff * v
                for d in 0..head_dim {
                    output[y_offset + d] = y_slice[d] - coeff * v_slice[d];
                }
            }
        }
    }
}

/// Backward for XSA.
/// grad_y, grad_v computed from grad_output (same shape as output).
pub fn xsa_backward(
    y: &[f32],
    v: &[f32],
    grad_output: &[f32],
    grad_y: &mut [f32],
    grad_v: &mut [f32],
    tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let group = num_heads / num_kv_heads;

    // Zero grad_v first (accumulated from multiple heads in group)
    grad_v.iter_mut().for_each(|x| *x = 0.0);

    for t in 0..tokens {
        for kv_h in 0..num_kv_heads {
            let v_offset = (t * num_kv_heads + kv_h) * head_dim;
            let v_slice = &v[v_offset..v_offset + head_dim];
            let v_norm_sq: f32 = v_slice.iter().map(|&x| x * x).sum::<f32>() + 1e-8;

            for g in 0..group {
                let h = kv_h * group + g;
                let y_offset = (t * num_heads + h) * head_dim;
                let y_slice = &y[y_offset..y_offset + head_dim];
                let go_slice = &grad_output[y_offset..y_offset + head_dim];

                let y_dot_v: f32 = y_slice
                    .iter()
                    .zip(v_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let coeff = y_dot_v / v_norm_sq;

                // grad_y = grad_out - (grad_out^T v / ||v||²) * v
                let go_dot_v: f32 = go_slice
                    .iter()
                    .zip(v_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let go_coeff = go_dot_v / v_norm_sq;

                for d in 0..head_dim {
                    grad_y[y_offset + d] = go_slice[d] - go_coeff * v_slice[d];
                }

                // grad_v = -coeff*go - (go·v / ||v||²)*y
                //          + (2*(y·v)*(go·v) / ||v||⁴)*v
                for d in 0..head_dim {
                    grad_v[v_offset + d] += -coeff * go_slice[d]
                        - (go_dot_v / v_norm_sq) * y_slice[d]
                        + (2.0 * y_dot_v * go_dot_v / (v_norm_sq * v_norm_sq)) * v_slice[d];
                }
            }
        }
    }
}

/// Reference custom SDPA+XSA kernel.
///
/// This is intentionally written as one fused semantic kernel rather than
/// `sdpa_forward` followed by `xsa_forward`: the self-value projection is
/// applied while the attention output is still local to the query position.
/// It is the CPU parity target for a future CUDA implementation.
#[allow(clippy::too_many_arguments)]
pub fn xsa_inside_sdpa_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
) {
    let tokens = batch * seq_len;
    let group = num_heads / num_kv_heads;
    assert_eq!(q.len(), tokens * num_heads * head_dim);
    assert_eq!(k.len(), tokens * num_kv_heads * head_dim);
    assert_eq!(v.len(), tokens * num_kv_heads * head_dim);
    assert_eq!(output.len(), q.len());

    let mut scores = vec![0.0f32; seq_len];
    for b in 0..batch {
        for t in 0..seq_len {
            let tok = b * seq_len + t;
            for kv_h in 0..num_kv_heads {
                let v_self_base = (tok * num_kv_heads + kv_h) * head_dim;
                let v_self = &v[v_self_base..v_self_base + head_dim];
                let v_norm_sq = v_self.iter().map(|x| x * x).sum::<f32>() + 1e-8;
                for g in 0..group {
                    let h = kv_h * group + g;
                    let q_base = (tok * num_heads + h) * head_dim;
                    let q_slice = &q[q_base..q_base + head_dim];

                    let mut max_score = f32::NEG_INFINITY;
                    for s in 0..=t {
                        let key_tok = b * seq_len + s;
                        let k_base = (key_tok * num_kv_heads + kv_h) * head_dim;
                        let score = dot(q_slice, &k[k_base..k_base + head_dim]) * softmax_scale;
                        scores[s] = score;
                        max_score = max_score.max(score);
                    }
                    let mut denom = 0.0f32;
                    for score in scores.iter_mut().take(t + 1) {
                        *score = (*score - max_score).exp();
                        denom += *score;
                    }

                    let out_base = (tok * num_heads + h) * head_dim;
                    for d in 0..head_dim {
                        let mut y = 0.0f32;
                        for s in 0..=t {
                            let value_tok = b * seq_len + s;
                            let v_base = (value_tok * num_kv_heads + kv_h) * head_dim;
                            y += (scores[s] / denom) * v[v_base + d];
                        }
                        output[out_base + d] = y;
                    }
                    let y_dot_v = dot(&output[out_base..out_base + head_dim], v_self);
                    let coeff = y_dot_v / v_norm_sq;
                    for d in 0..head_dim {
                        output[out_base + d] -= coeff * v_self[d];
                    }
                }
            }
        }
    }
}

/// Backward for [`xsa_inside_sdpa_forward`].
///
/// The derivative preserves exact SDPA gradients and the direct gradient from
/// the XSA self-value projection into the same V tensor used by attention.
#[allow(clippy::too_many_arguments)]
pub fn xsa_inside_sdpa_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_output: &[f32],
    grad_q: &mut [f32],
    grad_k: &mut [f32],
    grad_v: &mut [f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
) {
    let tokens = batch * seq_len;
    let group = num_heads / num_kv_heads;
    assert_eq!(q.len(), tokens * num_heads * head_dim);
    assert_eq!(k.len(), tokens * num_kv_heads * head_dim);
    assert_eq!(v.len(), tokens * num_kv_heads * head_dim);
    assert_eq!(grad_output.len(), q.len());
    assert_eq!(grad_q.len(), q.len());
    assert_eq!(grad_k.len(), k.len());
    assert_eq!(grad_v.len(), v.len());
    grad_q.fill(0.0);
    grad_k.fill(0.0);
    grad_v.fill(0.0);

    let mut scores = vec![0.0f32; seq_len];
    let mut probs = vec![0.0f32; seq_len];
    let mut y = vec![0.0f32; head_dim];
    let mut grad_y = vec![0.0f32; head_dim];

    for b in 0..batch {
        for t in 0..seq_len {
            let tok = b * seq_len + t;
            for kv_h in 0..num_kv_heads {
                let v_self_base = (tok * num_kv_heads + kv_h) * head_dim;
                let v_self = &v[v_self_base..v_self_base + head_dim];
                let v_norm_sq = v_self.iter().map(|x| x * x).sum::<f32>() + 1e-8;
                for g in 0..group {
                    let h = kv_h * group + g;
                    let q_base = (tok * num_heads + h) * head_dim;
                    let q_slice = &q[q_base..q_base + head_dim];

                    let mut max_score = f32::NEG_INFINITY;
                    for s in 0..=t {
                        let key_tok = b * seq_len + s;
                        let k_base = (key_tok * num_kv_heads + kv_h) * head_dim;
                        let score = dot(q_slice, &k[k_base..k_base + head_dim]) * softmax_scale;
                        scores[s] = score;
                        max_score = max_score.max(score);
                    }
                    let mut denom = 0.0f32;
                    for s in 0..=t {
                        probs[s] = (scores[s] - max_score).exp();
                        denom += probs[s];
                    }
                    for prob in probs.iter_mut().take(t + 1) {
                        *prob /= denom;
                    }

                    y.fill(0.0);
                    for s in 0..=t {
                        let value_tok = b * seq_len + s;
                        let v_base = (value_tok * num_kv_heads + kv_h) * head_dim;
                        for d in 0..head_dim {
                            y[d] += probs[s] * v[v_base + d];
                        }
                    }

                    let go_base = (tok * num_heads + h) * head_dim;
                    let go = &grad_output[go_base..go_base + head_dim];
                    let y_dot_v = dot(&y, v_self);
                    let coeff = y_dot_v / v_norm_sq;
                    let go_dot_v = dot(go, v_self);
                    let go_coeff = go_dot_v / v_norm_sq;

                    for d in 0..head_dim {
                        grad_y[d] = go[d] - go_coeff * v_self[d];
                        grad_v[v_self_base + d] += -coeff * go[d] - go_coeff * y[d]
                            + (2.0 * y_dot_v * go_dot_v / (v_norm_sq * v_norm_sq)) * v_self[d];
                    }

                    let mut grad_prob_weighted_sum = 0.0f32;
                    for s in 0..=t {
                        let value_tok = b * seq_len + s;
                        let v_base = (value_tok * num_kv_heads + kv_h) * head_dim;
                        let grad_prob = dot(&grad_y, &v[v_base..v_base + head_dim]);
                        scores[s] = grad_prob;
                        grad_prob_weighted_sum += probs[s] * grad_prob;
                        for d in 0..head_dim {
                            grad_v[v_base + d] += probs[s] * grad_y[d];
                        }
                    }
                    for s in 0..=t {
                        let value_tok = b * seq_len + s;
                        let k_base = (value_tok * num_kv_heads + kv_h) * head_dim;
                        let grad_pre_softmax = probs[s] * (scores[s] - grad_prob_weighted_sum);
                        for d in 0..head_dim {
                            grad_q[q_base + d] += grad_pre_softmax * softmax_scale * k[k_base + d];
                            grad_k[k_base + d] += grad_pre_softmax * softmax_scale * q_slice[d];
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xsa_removes_self_component() {
        // y = [1, 0], v = [1, 0] → z should be [0, 0] (y is entirely along v)
        let y = vec![1.0f32, 0.0];
        let v = vec![1.0f32, 0.0];
        let mut out = vec![0.0; 2];
        xsa_forward(&y, &v, &mut out, 1, 1, 1, 2);
        assert!(out[0].abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
    }

    #[test]
    fn test_xsa_preserves_orthogonal() {
        // y = [0, 1], v = [1, 0] → z should be [0, 1] (y is orthogonal to v)
        let y = vec![0.0f32, 1.0];
        let v = vec![1.0f32, 0.0];
        let mut out = vec![0.0; 2];
        xsa_forward(&y, &v, &mut out, 1, 1, 1, 2);
        assert!(out[0].abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_xsa_gqa() {
        // 2 heads, 1 KV head (group=2)
        // Both heads share the same v, each gets projected independently
        let y = vec![
            1.0, 0.0, // head 0
            0.5, 0.5, // head 1
        ];
        let v = vec![1.0, 0.0]; // single KV head
        let mut out = vec![0.0; 4];
        xsa_forward(&y, &v, &mut out, 1, 2, 1, 2);

        // head 0: y=[1,0], v=[1,0] → proj out → [0, 0]
        assert!(out[0].abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        // head 1: y=[0.5, 0.5], v=[1,0] → remove v-component → [0, 0.5]
        assert!(out[2].abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_xsa_backward_numerical() {
        let tokens = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 3;
        let y_len = tokens * num_heads * head_dim;
        let v_len = tokens * num_kv_heads * head_dim;
        let y: Vec<f32> = (0..y_len)
            .map(|i| 0.07 * ((i * 5 + 3) % 17) as f32 - 0.5)
            .collect();
        let v: Vec<f32> = (0..v_len)
            .map(|i| 0.05 * ((i * 7 + 2) % 19) as f32 - 0.4)
            .collect();
        let grad_out: Vec<f32> = (0..y_len)
            .map(|i| 0.03 * ((i * 11 + 1) % 13) as f32 - 0.2)
            .collect();

        let mut grad_y = vec![0.0; y_len];
        let mut grad_v = vec![0.0; v_len];
        xsa_backward(
            &y,
            &v,
            &grad_out,
            &mut grad_y,
            &mut grad_v,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let loss = |yy: &[f32], vv: &[f32]| -> f32 {
            let mut out = vec![0.0; y_len];
            xsa_forward(yy, vv, &mut out, tokens, num_heads, num_kv_heads, head_dim);
            out.iter().zip(grad_out.iter()).map(|(&a, &b)| a * b).sum()
        };
        let eps = 1e-3;
        for &idx in &[0usize, 5, y_len - 1] {
            let mut yp = y.clone();
            let mut ym = y.clone();
            yp[idx] += eps;
            ym[idx] -= eps;
            let numerical = (loss(&yp, &v) - loss(&ym, &v)) / (2.0 * eps);
            let diff = (grad_y[idx] - numerical).abs();
            assert!(
                diff < 2e-3,
                "grad_y[{idx}] analytical={} numerical={} diff={diff}",
                grad_y[idx],
                numerical
            );
        }
        for &idx in &[0usize, 4, v_len - 1] {
            let mut vp = v.clone();
            let mut vm = v.clone();
            vp[idx] += eps;
            vm[idx] -= eps;
            let numerical = (loss(&y, &vp) - loss(&y, &vm)) / (2.0 * eps);
            let diff = (grad_v[idx] - numerical).abs();
            assert!(
                diff < 2e-3,
                "grad_v[{idx}] analytical={} numerical={} diff={diff}",
                grad_v[idx],
                numerical
            );
        }
    }

    fn reference_causal_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        softmax_scale: f32,
    ) {
        let group = num_heads / num_kv_heads;
        let mut scores = vec![0.0f32; seq_len];
        for b in 0..batch {
            for t in 0..seq_len {
                let tok = b * seq_len + t;
                for kv_h in 0..num_kv_heads {
                    for g in 0..group {
                        let h = kv_h * group + g;
                        let q_base = (tok * num_heads + h) * head_dim;
                        let q_slice = &q[q_base..q_base + head_dim];
                        let mut max_score = f32::NEG_INFINITY;
                        for s in 0..=t {
                            let key_tok = b * seq_len + s;
                            let k_base = (key_tok * num_kv_heads + kv_h) * head_dim;
                            let score = dot(q_slice, &k[k_base..k_base + head_dim]) * softmax_scale;
                            scores[s] = score;
                            max_score = max_score.max(score);
                        }
                        let mut denom = 0.0f32;
                        for score in scores.iter_mut().take(t + 1) {
                            *score = (*score - max_score).exp();
                            denom += *score;
                        }
                        let out_base = (tok * num_heads + h) * head_dim;
                        for d in 0..head_dim {
                            let mut acc = 0.0;
                            for s in 0..=t {
                                let value_tok = b * seq_len + s;
                                let v_base = (value_tok * num_kv_heads + kv_h) * head_dim;
                                acc += scores[s] / denom * v[v_base + d];
                            }
                            out[out_base + d] = acc;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_xsa_inside_sdpa_forward_matches_separate_reference() {
        let batch = 2;
        let seq_len = 3;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 3;
        let tokens = batch * seq_len;
        let q_len = tokens * num_heads * head_dim;
        let kv_len = tokens * num_kv_heads * head_dim;
        let q: Vec<f32> = (0..q_len)
            .map(|i| ((i * 5 + 1) % 23) as f32 * 0.021 - 0.2)
            .collect();
        let k: Vec<f32> = (0..kv_len)
            .map(|i| ((i * 7 + 3) % 19) as f32 * 0.019 - 0.17)
            .collect();
        let v: Vec<f32> = (0..kv_len)
            .map(|i| ((i * 11 + 2) % 29) as f32 * 0.017 - 0.23)
            .collect();
        let mut sdpa = vec![0.0; q_len];
        reference_causal_sdpa(
            &q,
            &k,
            &v,
            &mut sdpa,
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            0.5,
        );
        let mut separate = vec![0.0; q_len];
        xsa_forward(
            &sdpa,
            &v,
            &mut separate,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        let mut fused = vec![0.0; q_len];
        xsa_inside_sdpa_forward(
            &q,
            &k,
            &v,
            &mut fused,
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            0.5,
        );
        for (idx, (&a, &b)) in separate.iter().zip(fused.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "inside-SDPA XSA mismatch at {idx}: separate={a} fused={b}"
            );
        }
    }

    #[test]
    fn test_xsa_inside_sdpa_backward_numerical() {
        let batch = 1;
        let seq_len = 3;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 3;
        let tokens = batch * seq_len;
        let q_len = tokens * num_heads * head_dim;
        let kv_len = tokens * num_kv_heads * head_dim;
        let q: Vec<f32> = (0..q_len)
            .map(|i| ((i * 3 + 1) % 17) as f32 * 0.017 - 0.11)
            .collect();
        let k: Vec<f32> = (0..kv_len)
            .map(|i| ((i * 5 + 2) % 19) as f32 * 0.019 - 0.13)
            .collect();
        let v: Vec<f32> = (0..kv_len)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.013 - 0.09)
            .collect();
        let grad_out: Vec<f32> = (0..q_len)
            .map(|i| ((i * 11 + 4) % 13) as f32 * 0.023 - 0.07)
            .collect();
        let mut grad_q = vec![0.0; q_len];
        let mut grad_k = vec![0.0; kv_len];
        let mut grad_v = vec![0.0; kv_len];
        xsa_inside_sdpa_backward(
            &q,
            &k,
            &v,
            &grad_out,
            &mut grad_q,
            &mut grad_k,
            &mut grad_v,
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            0.5,
        );

        let loss = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            let mut out = vec![0.0; q_len];
            xsa_inside_sdpa_forward(
                qq,
                kk,
                vv,
                &mut out,
                batch,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0.5,
            );
            out.iter().zip(grad_out.iter()).map(|(&a, &b)| a * b).sum()
        };
        let eps = 1e-3;
        for &idx in &[0usize, 4, q_len - 1] {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let numerical = (loss(&plus, &k, &v) - loss(&minus, &k, &v)) / (2.0 * eps);
            assert!(
                (grad_q[idx] - numerical).abs() < 3e-3,
                "grad_q[{idx}] analytical={} numerical={}",
                grad_q[idx],
                numerical
            );
        }
        for &idx in &[0usize, 3, kv_len - 1] {
            let mut plus = k.clone();
            let mut minus = k.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let numerical = (loss(&q, &plus, &v) - loss(&q, &minus, &v)) / (2.0 * eps);
            assert!(
                (grad_k[idx] - numerical).abs() < 3e-3,
                "grad_k[{idx}] analytical={} numerical={}",
                grad_k[idx],
                numerical
            );
        }
        for &idx in &[0usize, 2, kv_len - 1] {
            let mut plus = v.clone();
            let mut minus = v.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let numerical = (loss(&q, &k, &plus) - loss(&q, &k, &minus)) / (2.0 * eps);
            assert!(
                (grad_v[idx] - numerical).abs() < 3e-3,
                "grad_v[{idx}] analytical={} numerical={}",
                grad_v[idx],
                numerical
            );
        }
    }

    #[test]
    fn test_adjacent_sparse_gate_then_xsa_matches_composite_reference() {
        let tokens = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 3;
        let model_dim = num_heads * head_dim;
        let gate_width = 5;
        let scale = 0.75;
        let attn_raw: Vec<f32> = (0..tokens * num_heads * head_dim)
            .map(|i| ((i * 3 + 1) % 17) as f32 * 0.02 - 0.12)
            .collect();
        let v: Vec<f32> = (0..tokens * num_kv_heads * head_dim)
            .map(|i| ((i * 5 + 2) % 19) as f32 * 0.015 - 0.08)
            .collect();
        let attn_norm_out: Vec<f32> = (0..tokens * model_dim)
            .map(|i| ((i * 7 + 4) % 23) as f32 * 0.01 - 0.1)
            .collect();
        let gate_weight: Vec<f32> = (0..num_heads * gate_width)
            .map(|i| ((i * 11 + 3) % 13) as f32 * 0.03 - 0.18)
            .collect();

        let mut gated = vec![0.0; attn_raw.len()];
        for tok in 0..tokens {
            let gate_input = &attn_norm_out[tok * model_dim..tok * model_dim + gate_width];
            for head in 0..num_heads {
                let weight = &gate_weight[head * gate_width..(head + 1) * gate_width];
                let score = weight
                    .iter()
                    .zip(gate_input)
                    .map(|(w, x)| w * x)
                    .sum::<f32>();
                let gate = 1.0 / (1.0 + (-(scale * score)).exp());
                let base = (tok * num_heads + head) * head_dim;
                for dim in 0..head_dim {
                    gated[base + dim] = attn_raw[base + dim] * gate;
                }
            }
        }
        let mut separate = vec![0.0; attn_raw.len()];
        xsa_forward(
            &gated,
            &v,
            &mut separate,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let mut composite = vec![0.0; attn_raw.len()];
        let group = num_heads / num_kv_heads;
        for tok in 0..tokens {
            let gate_input = &attn_norm_out[tok * model_dim..tok * model_dim + gate_width];
            for kv_h in 0..num_kv_heads {
                let v_offset = (tok * num_kv_heads + kv_h) * head_dim;
                let v_slice = &v[v_offset..v_offset + head_dim];
                let v_norm_sq = v_slice.iter().map(|x| x * x).sum::<f32>() + 1e-8;
                for g in 0..group {
                    let head = kv_h * group + g;
                    let weight = &gate_weight[head * gate_width..(head + 1) * gate_width];
                    let score = weight
                        .iter()
                        .zip(gate_input)
                        .map(|(w, x)| w * x)
                        .sum::<f32>();
                    let gate = 1.0 / (1.0 + (-(scale * score)).exp());
                    let base = (tok * num_heads + head) * head_dim;
                    let dot = (0..head_dim)
                        .map(|d| attn_raw[base + d] * gate * v_slice[d])
                        .sum::<f32>();
                    let coeff = dot / v_norm_sq;
                    for d in 0..head_dim {
                        composite[base + d] = attn_raw[base + d] * gate - coeff * v_slice[d];
                    }
                }
            }
        }

        for (idx, (a, b)) in separate.iter().zip(composite.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "adjacent SparseAttnGate+XSA mismatch at {idx}: {a} vs {b}"
            );
        }
    }
}
