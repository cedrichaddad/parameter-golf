/// Fused cross-entropy loss with logit softcap.
///
/// Computes: loss = -log(softmax(softcap(logits))[target])
/// where softcap(x) = cap * tanh(x / cap)
///
/// The fused implementation avoids materializing the full softmax output.
/// With vocab=1024 this saves modest memory but eliminates a global memory
/// round-trip.

/// Forward: compute per-token cross-entropy loss with softcap.
/// logits: [num_tokens, vocab_size]
/// targets: [num_tokens] (token IDs)
/// Returns per-token losses.
pub fn cross_entropy_forward(
    logits: &[f32],
    targets: &[u32],
    losses: &mut [f32],
    vocab_size: usize,
    softcap: f32,
) {
    cross_entropy_forward_asym(logits, targets, losses, vocab_size, softcap, softcap);
}

pub fn cross_entropy_forward_asym(
    logits: &[f32],
    targets: &[u32],
    losses: &mut [f32],
    vocab_size: usize,
    softcap_pos: f32,
    softcap_neg: f32,
) {
    let num_tokens = targets.len();
    let cap = |v: f32| {
        let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
        if softcap > 0.0 {
            softcap * (v / softcap).tanh()
        } else {
            v
        }
    };

    for t in 0..num_tokens {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];
        let target = targets[t] as usize;

        // Apply softcap: cap * tanh(x / cap)
        // Find max for numerical stability (after softcap)
        let mut max_val = f32::NEG_INFINITY;
        for &v in row {
            let capped = cap(v);
            if capped > max_val {
                max_val = capped;
            }
        }

        // Compute log-sum-exp
        let mut sum_exp = 0.0f32;
        for &v in row {
            let capped = cap(v);
            sum_exp += (capped - max_val).exp();
        }
        let log_sum_exp = max_val + sum_exp.ln();

        // Loss = -log(softmax[target]) = -(capped_target - log_sum_exp)
        let capped_target = cap(row[target]);
        losses[t] = log_sum_exp - capped_target;
    }
}

/// Backward: compute gradient w.r.t. logits.
/// grad_logits: [num_tokens, vocab_size]
/// grad_loss: scalar multiplier (typically 1/num_tokens for mean reduction)
pub fn cross_entropy_backward(
    logits: &[f32],
    targets: &[u32],
    grad_logits: &mut [f32],
    vocab_size: usize,
    softcap: f32,
    grad_loss: f32,
) {
    cross_entropy_backward_asym(
        logits,
        targets,
        grad_logits,
        vocab_size,
        softcap,
        softcap,
        grad_loss,
    );
}

pub fn cross_entropy_backward_asym(
    logits: &[f32],
    targets: &[u32],
    grad_logits: &mut [f32],
    vocab_size: usize,
    softcap_pos: f32,
    softcap_neg: f32,
    grad_loss: f32,
) {
    let num_tokens = targets.len();
    let mut exps = vec![0.0f32; vocab_size]; // pre-allocated, reused across tokens
    let cap = |v: f32| {
        let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
        if softcap > 0.0 {
            softcap * (v / softcap).tanh()
        } else {
            v
        }
    };
    let cap_deriv = |v: f32| {
        let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
        if softcap > 0.0 {
            let t_val = (v / softcap).tanh();
            1.0 - t_val * t_val
        } else {
            1.0
        }
    };

    for t in 0..num_tokens {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];
        let target = targets[t] as usize;

        // Compute softmax of capped logits
        let mut max_val = f32::NEG_INFINITY;
        for &v in row {
            let capped = cap(v);
            if capped > max_val {
                max_val = capped;
            }
        }

        let mut sum_exp = 0.0f32;
        for (i, &v) in row.iter().enumerate() {
            let capped = cap(v);
            exps[i] = (capped - max_val).exp();
            sum_exp += exps[i];
        }

        // grad_logits = grad_loss * (softmax - one_hot) * d_softcap/d_logits
        for i in 0..vocab_size {
            let prob = exps[i] / sum_exp;
            let one_hot = if i == target { 1.0 } else { 0.0 };
            // d_softcap/d_x = 1 - tanh²(x/cap)
            grad_logits[offset + i] = grad_loss * (prob - one_hot) * cap_deriv(row[i]);
        }
    }
}

/// Compute mean loss from per-token losses.
pub fn mean_loss(losses: &[f32]) -> f32 {
    if losses.is_empty() {
        return 0.0;
    }
    losses.iter().sum::<f32>() / losses.len() as f32
}

/// Exact output projection + softcapped CE without persistent full logits.
///
/// `hidden` is `[m, d]`, `weight` is `[vocab_size, d]`, and `targets` is `[m]`.
/// The implementation streams vocabulary tiles, recomputes dot products as
/// needed, and writes only per-row losses. It is a CPU reference for
/// KernelForge/tiled GPU implementations.
pub fn tiled_output_cross_entropy_forward(
    hidden: &[f32],
    weight: &[f32],
    targets: &[u32],
    losses: &mut [f32],
    m: usize,
    d: usize,
    vocab_size: usize,
    tile_vocab: usize,
    softcap: f32,
) {
    tiled_output_cross_entropy_forward_asym(
        hidden, weight, targets, losses, m, d, vocab_size, tile_vocab, softcap, softcap,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn tiled_output_cross_entropy_forward_asym(
    hidden: &[f32],
    weight: &[f32],
    targets: &[u32],
    losses: &mut [f32],
    m: usize,
    d: usize,
    vocab_size: usize,
    tile_vocab: usize,
    softcap_pos: f32,
    softcap_neg: f32,
) {
    assert_eq!(hidden.len(), m * d);
    assert_eq!(weight.len(), vocab_size * d);
    assert_eq!(targets.len(), m);
    assert_eq!(losses.len(), m);
    let tile_vocab = tile_vocab.max(1).min(vocab_size.max(1));

    for row in 0..m {
        let target = targets[row] as usize;
        assert!(target < vocab_size, "target out of vocabulary: {target}");
        let h = &hidden[row * d..(row + 1) * d];
        let mut row_max = f32::NEG_INFINITY;
        let mut target_logit = 0.0f32;

        for tile_start in (0..vocab_size).step_by(tile_vocab) {
            let tile_end = (tile_start + tile_vocab).min(vocab_size);
            for v in tile_start..tile_end {
                let logit = dot(h, &weight[v * d..(v + 1) * d]);
                let capped = softcap_value(logit, softcap_pos, softcap_neg);
                row_max = row_max.max(capped);
                if v == target {
                    target_logit = capped;
                }
            }
        }

        let mut row_sum = 0.0f32;
        for tile_start in (0..vocab_size).step_by(tile_vocab) {
            let tile_end = (tile_start + tile_vocab).min(vocab_size);
            for v in tile_start..tile_end {
                let logit = dot(h, &weight[v * d..(v + 1) * d]);
                let capped = softcap_value(logit, softcap_pos, softcap_neg);
                row_sum += (capped - row_max).exp();
            }
        }
        losses[row] = row_max + row_sum.ln() - target_logit;
    }
}

/// Exact tiled output projection + softcapped CE backward.
///
/// Fills `d_hidden` `[m, d]` and `d_weight` `[vocab_size, d]` with gradients
/// for `loss_scale * sum_i CE_i`. No persistent `[m, vocab_size]` logits or
/// grad-logits tensor is allocated.
#[allow(clippy::too_many_arguments)]
pub fn tiled_output_cross_entropy_backward(
    hidden: &[f32],
    weight: &[f32],
    targets: &[u32],
    d_hidden: &mut [f32],
    d_weight: &mut [f32],
    m: usize,
    d: usize,
    vocab_size: usize,
    tile_vocab: usize,
    softcap: f32,
    loss_scale: f32,
) {
    tiled_output_cross_entropy_backward_asym(
        hidden, weight, targets, d_hidden, d_weight, m, d, vocab_size, tile_vocab, softcap,
        softcap, loss_scale,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn tiled_output_cross_entropy_backward_asym(
    hidden: &[f32],
    weight: &[f32],
    targets: &[u32],
    d_hidden: &mut [f32],
    d_weight: &mut [f32],
    m: usize,
    d: usize,
    vocab_size: usize,
    tile_vocab: usize,
    softcap_pos: f32,
    softcap_neg: f32,
    loss_scale: f32,
) {
    assert_eq!(hidden.len(), m * d);
    assert_eq!(weight.len(), vocab_size * d);
    assert_eq!(targets.len(), m);
    assert_eq!(d_hidden.len(), m * d);
    assert_eq!(d_weight.len(), vocab_size * d);
    let tile_vocab = tile_vocab.max(1).min(vocab_size.max(1));
    d_hidden.fill(0.0);
    d_weight.fill(0.0);

    for row in 0..m {
        let target = targets[row] as usize;
        assert!(target < vocab_size, "target out of vocabulary: {target}");
        let h = &hidden[row * d..(row + 1) * d];
        let mut row_max = f32::NEG_INFINITY;

        for tile_start in (0..vocab_size).step_by(tile_vocab) {
            let tile_end = (tile_start + tile_vocab).min(vocab_size);
            for v in tile_start..tile_end {
                let logit = dot(h, &weight[v * d..(v + 1) * d]);
                row_max = row_max.max(softcap_value(logit, softcap_pos, softcap_neg));
            }
        }

        let mut row_sum = 0.0f32;
        for tile_start in (0..vocab_size).step_by(tile_vocab) {
            let tile_end = (tile_start + tile_vocab).min(vocab_size);
            for v in tile_start..tile_end {
                let logit = dot(h, &weight[v * d..(v + 1) * d]);
                let capped = softcap_value(logit, softcap_pos, softcap_neg);
                row_sum += (capped - row_max).exp();
            }
        }

        for tile_start in (0..vocab_size).step_by(tile_vocab) {
            let tile_end = (tile_start + tile_vocab).min(vocab_size);
            for v in tile_start..tile_end {
                let w = &weight[v * d..(v + 1) * d];
                let logit = dot(h, w);
                let capped = softcap_value(logit, softcap_pos, softcap_neg);
                let prob = (capped - row_max).exp() / row_sum;
                let one_hot = if v == target { 1.0 } else { 0.0 };
                let grad_logit = loss_scale
                    * (prob - one_hot)
                    * softcap_derivative(logit, softcap_pos, softcap_neg);

                for col in 0..d {
                    d_hidden[row * d + col] += grad_logit * w[col];
                    d_weight[v * d + col] += grad_logit * h[col];
                }
            }
        }
    }
}

fn softcap_value(v: f32, softcap_pos: f32, softcap_neg: f32) -> f32 {
    let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
    if softcap > 0.0 {
        softcap * (v / softcap).tanh()
    } else {
        v
    }
}

fn softcap_derivative(v: f32, softcap_pos: f32, softcap_neg: f32) -> f32 {
    let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
    if softcap > 0.0 {
        let t_val = (v / softcap).tanh();
        1.0 - t_val * t_val
    } else {
        1.0
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_basic() {
        // vocab=4, one token, target=2
        // logits that strongly predict class 2
        let logits = vec![0.0f32, 0.0, 10.0, 0.0];
        let targets = vec![2u32];
        let mut losses = vec![0.0f32];

        cross_entropy_forward(&logits, &targets, &mut losses, 4, 30.0);
        // With softcap=30, tanh(10/30) ≈ 0.321, capped ≈ 9.64
        // Should be low loss since target has highest logit
        assert!(losses[0] < 1.0, "loss too high: {}", losses[0]);
        assert!(losses[0] > 0.0, "loss should be positive");
    }

    #[test]
    fn test_cross_entropy_uniform() {
        // All logits equal → loss = ln(vocab_size)
        let vocab = 1024;
        let logits = vec![0.0f32; vocab];
        let targets = vec![0u32];
        let mut losses = vec![0.0f32];

        cross_entropy_forward(&logits, &targets, &mut losses, vocab, 30.0);
        let expected = (vocab as f32).ln();
        assert!(
            (losses[0] - expected).abs() < 0.01,
            "expected ~{}, got {}",
            expected,
            losses[0]
        );
    }

    #[test]
    fn test_cross_entropy_gradient() {
        // Numerical gradient check
        let vocab = 8;
        let logits = vec![1.0f32, -0.5, 2.0, 0.3, -1.0, 0.5, 1.5, -0.2];
        let targets = vec![3u32];
        let softcap = 30.0;

        let mut grad = vec![0.0f32; vocab];
        cross_entropy_backward(&logits, &targets, &mut grad, vocab, softcap, 1.0);

        // Numerical gradient
        let eps = 1e-4;
        for i in 0..vocab {
            let mut logits_plus = logits.clone();
            let mut logits_minus = logits.clone();
            logits_plus[i] += eps;
            logits_minus[i] -= eps;

            let mut loss_plus = vec![0.0f32];
            let mut loss_minus = vec![0.0f32];
            cross_entropy_forward(&logits_plus, &targets, &mut loss_plus, vocab, softcap);
            cross_entropy_forward(&logits_minus, &targets, &mut loss_minus, vocab, softcap);

            let numerical = (loss_plus[0] - loss_minus[0]) / (2.0 * eps);
            let tol = 2e-2 * grad[i].abs().max(numerical.abs()).max(1e-3);
            assert!(
                (grad[i] - numerical).abs() < tol,
                "grad mismatch at {}: analytical={}, numerical={}, diff={}",
                i,
                grad[i],
                numerical,
                (grad[i] - numerical).abs()
            );
        }
    }

    #[test]
    fn test_asymmetric_softcap_changes_negative_logits() {
        let vocab = 4;
        let logits = vec![-40.0f32, -2.0, 1.0, 4.0];
        let targets = vec![0u32];
        let mut symmetric = vec![0.0f32];
        let mut asymmetric = vec![0.0f32];

        cross_entropy_forward(&logits, &targets, &mut symmetric, vocab, 30.0);
        cross_entropy_forward_asym(&logits, &targets, &mut asymmetric, vocab, 30.0, 34.0);

        assert!(
            (symmetric[0] - asymmetric[0]).abs() > 1e-3,
            "asymmetric negative softcap should affect loss"
        );
    }

    #[test]
    fn tiled_output_ce_forward_matches_full_logits() {
        let m = 5;
        let d = 7;
        let vocab = 13;
        let hidden = deterministic_values(m * d, 0.17);
        let weight = deterministic_values(vocab * d, 0.31);
        let targets = vec![0, 3, 7, 12, 5];
        let softcap_pos = 19.0;
        let softcap_neg = 23.0;

        let logits = output_logits(&hidden, &weight, m, d, vocab);
        let mut expected = vec![0.0f32; m];
        cross_entropy_forward_asym(
            &logits,
            &targets,
            &mut expected,
            vocab,
            softcap_pos,
            softcap_neg,
        );

        for tile_vocab in [1, 2, 5, 64] {
            let mut got = vec![0.0f32; m];
            tiled_output_cross_entropy_forward_asym(
                &hidden,
                &weight,
                &targets,
                &mut got,
                m,
                d,
                vocab,
                tile_vocab,
                softcap_pos,
                softcap_neg,
            );
            assert_close_slice(&got, &expected, 2e-6, "tiled forward loss");
        }
    }

    #[test]
    fn tiled_output_ce_backward_matches_full_logits_chain_rule() {
        let m = 4;
        let d = 6;
        let vocab = 11;
        let hidden = deterministic_values(m * d, 0.23);
        let weight = deterministic_values(vocab * d, 0.41);
        let targets = vec![1, 10, 4, 7];
        let softcap_pos = 17.0;
        let softcap_neg = 29.0;
        let loss_scale = 0.25;

        let logits = output_logits(&hidden, &weight, m, d, vocab);
        let mut grad_logits = vec![0.0f32; m * vocab];
        cross_entropy_backward_asym(
            &logits,
            &targets,
            &mut grad_logits,
            vocab,
            softcap_pos,
            softcap_neg,
            loss_scale,
        );
        let expected_d_hidden = grad_logits_times_weight(&grad_logits, &weight, m, d, vocab);
        let expected_d_weight = grad_logits_t_times_hidden(&grad_logits, &hidden, m, d, vocab);

        for tile_vocab in [1, 3, 4, 32] {
            let mut got_d_hidden = vec![0.0f32; m * d];
            let mut got_d_weight = vec![0.0f32; vocab * d];
            tiled_output_cross_entropy_backward_asym(
                &hidden,
                &weight,
                &targets,
                &mut got_d_hidden,
                &mut got_d_weight,
                m,
                d,
                vocab,
                tile_vocab,
                softcap_pos,
                softcap_neg,
                loss_scale,
            );
            assert_close_slice(&got_d_hidden, &expected_d_hidden, 3e-6, "tiled d_hidden");
            assert_close_slice(&got_d_weight, &expected_d_weight, 3e-6, "tiled d_weight");
        }
    }

    fn deterministic_values(n: usize, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f32 + 1.0;
                0.7 * (x * phase).sin() + 0.3 * (x * phase * 0.37).cos()
            })
            .collect()
    }

    fn output_logits(hidden: &[f32], weight: &[f32], m: usize, d: usize, vocab: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; m * vocab];
        for row in 0..m {
            for v in 0..vocab {
                logits[row * vocab + v] =
                    dot(&hidden[row * d..(row + 1) * d], &weight[v * d..(v + 1) * d]);
            }
        }
        logits
    }

    fn grad_logits_times_weight(
        grad_logits: &[f32],
        weight: &[f32],
        m: usize,
        d: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * d];
        for row in 0..m {
            for v in 0..vocab {
                let g = grad_logits[row * vocab + v];
                for col in 0..d {
                    out[row * d + col] += g * weight[v * d + col];
                }
            }
        }
        out
    }

    fn grad_logits_t_times_hidden(
        grad_logits: &[f32],
        hidden: &[f32],
        m: usize,
        d: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; vocab * d];
        for row in 0..m {
            for v in 0..vocab {
                let g = grad_logits[row * vocab + v];
                for col in 0..d {
                    out[v * d + col] += g * hidden[row * d + col];
                }
            }
        }
        out
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
