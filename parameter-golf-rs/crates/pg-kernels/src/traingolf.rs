use pg_core::error::{PgError, PgResult};

#[derive(Clone, Copy, Debug)]
pub struct TrainGolfQuantizationConfig {
    pub bits: u8,
    pub block_size: usize,
    pub lambda: f32,
}

#[derive(Clone, Debug)]
pub struct TrainGolfRegularizerReport {
    pub projected: Vec<f32>,
    pub gradient: Vec<f32>,
    pub distance_sq: f64,
    pub distance_norm: f64,
    pub regularization_loss: f64,
}

pub fn traingolf_quantization_regularizer(
    weights: &[f32],
    config: TrainGolfQuantizationConfig,
) -> PgResult<TrainGolfRegularizerReport> {
    validate_config(config)?;
    let projected = traingolf_project_to_quant_grid(weights, config.bits, config.block_size)?;
    let mut gradient = vec![0.0f32; weights.len()];
    let mut distance_sq = 0.0f64;
    for (i, (&weight, &quantized)) in weights.iter().zip(&projected).enumerate() {
        let residual = weight - quantized;
        distance_sq += (residual as f64) * (residual as f64);
        gradient[i] = 2.0 * config.lambda * residual;
    }
    Ok(TrainGolfRegularizerReport {
        projected,
        gradient,
        distance_sq,
        distance_norm: distance_sq.sqrt(),
        regularization_loss: config.lambda as f64 * distance_sq,
    })
}

pub fn traingolf_project_to_quant_grid(
    weights: &[f32],
    bits: u8,
    block_size: usize,
) -> PgResult<Vec<f32>> {
    validate_bits(bits)?;
    let block_size = block_size.max(1);
    let qmax = signed_qmax(bits) as f32;
    let mut projected = vec![0.0f32; weights.len()];

    for block_start in (0..weights.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(weights.len());
        let max_abs = weights[block_start..block_end]
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let scale = max_abs / qmax;
        for i in block_start..block_end {
            let q = (weights[i] / scale).round().clamp(-qmax, qmax);
            projected[i] = q * scale;
        }
    }
    Ok(projected)
}

pub fn traingolf_export_gap_bound(
    gradient_norm: f64,
    quantization_distance_norm: f64,
    smoothness: f64,
) -> PgResult<f64> {
    if !gradient_norm.is_finite() || gradient_norm < 0.0 {
        return Err(PgError::InvalidOp(format!(
            "gradient_norm must be nonnegative finite, got {gradient_norm}"
        )));
    }
    if !quantization_distance_norm.is_finite() || quantization_distance_norm < 0.0 {
        return Err(PgError::InvalidOp(format!(
            "quantization_distance_norm must be nonnegative finite, got {quantization_distance_norm}"
        )));
    }
    if !smoothness.is_finite() || smoothness < 0.0 {
        return Err(PgError::InvalidOp(format!(
            "smoothness must be nonnegative finite, got {smoothness}"
        )));
    }
    Ok(gradient_norm * quantization_distance_norm
        + 0.5 * smoothness * quantization_distance_norm * quantization_distance_norm)
}

pub fn traingolf_stop_gradient_regularized_loss(
    base_loss: f64,
    regularization_loss: f64,
) -> PgResult<f64> {
    if !base_loss.is_finite() {
        return Err(PgError::InvalidOp(format!(
            "base_loss must be finite, got {base_loss}"
        )));
    }
    if !regularization_loss.is_finite() || regularization_loss < 0.0 {
        return Err(PgError::InvalidOp(format!(
            "regularization_loss must be nonnegative finite, got {regularization_loss}"
        )));
    }
    Ok(base_loss + regularization_loss)
}

fn validate_config(config: TrainGolfQuantizationConfig) -> PgResult<()> {
    validate_bits(config.bits)?;
    if !config.lambda.is_finite() || config.lambda < 0.0 {
        return Err(PgError::InvalidOp(format!(
            "lambda must be nonnegative finite, got {}",
            config.lambda
        )));
    }
    Ok(())
}

fn validate_bits(bits: u8) -> PgResult<()> {
    if (2..=8).contains(&bits) {
        Ok(())
    } else {
        Err(PgError::InvalidOp(format!(
            "bits must be in 2..=8, got {bits}"
        )))
    }
}

fn signed_qmax(bits: u8) -> i32 {
    (1i32 << (bits - 1)) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_regularizer_is_zero_on_grid_values() {
        let weights = vec![-1.0f32, -1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0];
        let report = traingolf_quantization_regularizer(
            &weights,
            TrainGolfQuantizationConfig {
                bits: 3,
                block_size: weights.len(),
                lambda: 0.7,
            },
        )
        .unwrap();

        assert_close_slice(&report.projected, &weights, 1e-6, "projected");
        assert_close_slice(
            &report.gradient,
            &vec![0.0; weights.len()],
            1e-6,
            "gradient",
        );
        assert!(report.distance_sq < 1e-12);
        assert!(report.regularization_loss < 1e-12);
    }

    #[test]
    fn regularizer_gradient_matches_stop_gradient_distance_objective() {
        let weights = vec![0.42f32, -0.17, 0.05, 0.31, -0.29];
        let lambda = 0.25;
        let report = traingolf_quantization_regularizer(
            &weights,
            TrainGolfQuantizationConfig {
                bits: 4,
                block_size: 3,
                lambda,
            },
        )
        .unwrap();

        for i in 0..weights.len() {
            let expected = 2.0 * lambda * (weights[i] - report.projected[i]);
            assert!(
                (report.gradient[i] - expected).abs() < 1e-6,
                "gradient mismatch at {i}: got {}, expected {expected}",
                report.gradient[i]
            );
        }
        assert!((report.regularization_loss - lambda as f64 * report.distance_sq).abs() < 1e-12);
    }

    #[test]
    fn stop_gradient_step_reduces_fixed_projection_distance() {
        let weights = vec![0.48f32, -0.18, 0.09, 0.37];
        let lambda = 0.5;
        let report = traingolf_quantization_regularizer(
            &weights,
            TrainGolfQuantizationConfig {
                bits: 3,
                block_size: weights.len(),
                lambda,
            },
        )
        .unwrap();
        let eta = 0.4;
        let stepped = weights
            .iter()
            .zip(&report.gradient)
            .map(|(weight, grad)| weight - eta * grad)
            .collect::<Vec<_>>();
        let before = squared_distance(&weights, &report.projected);
        let after = squared_distance(&stepped, &report.projected);
        assert!(
            after < before,
            "after {after} should be below before {before}"
        );
    }

    #[test]
    fn export_gap_bound_covers_quadratic_smooth_loss() {
        let weights = vec![0.42f32, -0.17, 0.05, 0.31, -0.29, 0.11];
        let smoothness = 1.7f64;
        let report = traingolf_quantization_regularizer(
            &weights,
            TrainGolfQuantizationConfig {
                bits: 3,
                block_size: 4,
                lambda: 0.2,
            },
        )
        .unwrap();

        let full_loss = quadratic_loss(&weights, smoothness);
        let exported_loss = quadratic_loss(&report.projected, smoothness);
        let grad_norm = smoothness * l2_norm(&weights);
        let bound =
            traingolf_export_gap_bound(grad_norm, report.distance_norm, smoothness).unwrap();
        assert!(
            exported_loss - full_loss <= bound + 1e-12,
            "gap {} exceeded bound {bound}",
            exported_loss - full_loss
        );
    }

    #[test]
    fn invalid_quantization_config_is_rejected() {
        let err = traingolf_quantization_regularizer(
            &[1.0, 2.0],
            TrainGolfQuantizationConfig {
                bits: 1,
                block_size: 1,
                lambda: 0.1,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("bits must be in 2..=8"));

        let err = traingolf_export_gap_bound(1.0, 1.0, -0.1).unwrap_err();
        assert!(err.to_string().contains("smoothness"));
    }

    fn quadratic_loss(weights: &[f32], smoothness: f64) -> f64 {
        0.5 * smoothness
            * weights
                .iter()
                .map(|value| (*value as f64).powi(2))
                .sum::<f64>()
    }

    fn l2_norm(values: &[f32]) -> f64 {
        values
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn squared_distance(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let diff = *x as f64 - *y as f64;
                diff * diff
            })
            .sum()
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
