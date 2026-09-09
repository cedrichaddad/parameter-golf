use pg_core::error::{PgError, PgResult};

#[derive(Clone, Debug)]
pub struct DeltaGolfLowRankDelta {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub singular_values: Vec<f32>,
}

pub fn deltagolf_weighted_low_rank_delta(
    full_update: &[f32],
    out_curvature_diag: &[f32],
    in_curvature_diag: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> PgResult<DeltaGolfLowRankDelta> {
    validate_weighted_delta_inputs(
        full_update,
        out_curvature_diag,
        in_curvature_diag,
        rows,
        cols,
    )?;
    let rank = rank.min(rows.min(cols));
    if rank == 0 {
        return Ok(DeltaGolfLowRankDelta {
            rows,
            cols,
            rank,
            a: Vec::new(),
            b: Vec::new(),
            singular_values: Vec::new(),
        });
    }

    let mut whitened = vec![0.0f64; rows * cols];
    for row in 0..rows {
        let out_sqrt = (out_curvature_diag[row] as f64).sqrt();
        for col in 0..cols {
            let in_sqrt = (in_curvature_diag[col] as f64).sqrt();
            whitened[row * cols + col] = full_update[row * cols + col] as f64 * out_sqrt * in_sqrt;
        }
    }

    let components = truncated_svd_components(&whitened, rows, cols, rank);
    let actual_rank = components.len();
    let mut a = vec![0.0f32; rows * actual_rank];
    let mut b = vec![0.0f32; actual_rank * cols];
    let mut singular_values = Vec::with_capacity(actual_rank);

    for (component_index, component) in components.iter().enumerate() {
        singular_values.push(component.sigma as f32);
        for row in 0..rows {
            let out_inv_sqrt = 1.0 / (out_curvature_diag[row] as f64).sqrt();
            a[row * actual_rank + component_index] =
                (component.u[row] * component.sigma * out_inv_sqrt) as f32;
        }
        for col in 0..cols {
            let in_inv_sqrt = 1.0 / (in_curvature_diag[col] as f64).sqrt();
            b[component_index * cols + col] = (component.v[col] * in_inv_sqrt) as f32;
        }
    }

    Ok(DeltaGolfLowRankDelta {
        rows,
        cols,
        rank: actual_rank,
        a,
        b,
        singular_values,
    })
}

pub fn deltagolf_materialize_delta(
    delta: &DeltaGolfLowRankDelta,
    output: &mut [f32],
) -> PgResult<()> {
    if output.len() != delta.rows * delta.cols {
        return Err(PgError::ShapeMismatch {
            expected: vec![delta.rows, delta.cols],
            got: vec![output.len()],
        });
    }
    output.fill(0.0);
    for row in 0..delta.rows {
        for k in 0..delta.rank {
            let a = delta.a[row * delta.rank + k];
            for col in 0..delta.cols {
                output[row * delta.cols + col] += a * delta.b[k * delta.cols + col];
            }
        }
    }
    Ok(())
}

pub fn deltagolf_weighted_error(
    candidate_delta: &[f32],
    full_update: &[f32],
    out_curvature_diag: &[f32],
    in_curvature_diag: &[f32],
    rows: usize,
    cols: usize,
) -> PgResult<f64> {
    validate_weighted_delta_inputs(
        full_update,
        out_curvature_diag,
        in_curvature_diag,
        rows,
        cols,
    )?;
    if candidate_delta.len() != rows * cols {
        return Err(PgError::ShapeMismatch {
            expected: vec![rows, cols],
            got: vec![candidate_delta.len()],
        });
    }
    let mut error = 0.0f64;
    for row in 0..rows {
        for col in 0..cols {
            let diff =
                candidate_delta[row * cols + col] as f64 - full_update[row * cols + col] as f64;
            error += out_curvature_diag[row] as f64 * diff * diff * in_curvature_diag[col] as f64;
        }
    }
    Ok(error)
}

fn validate_weighted_delta_inputs(
    full_update: &[f32],
    out_curvature_diag: &[f32],
    in_curvature_diag: &[f32],
    rows: usize,
    cols: usize,
) -> PgResult<()> {
    if full_update.len() != rows * cols {
        return Err(PgError::ShapeMismatch {
            expected: vec![rows, cols],
            got: vec![full_update.len()],
        });
    }
    if out_curvature_diag.len() != rows {
        return Err(PgError::ShapeMismatch {
            expected: vec![rows],
            got: vec![out_curvature_diag.len()],
        });
    }
    if in_curvature_diag.len() != cols {
        return Err(PgError::ShapeMismatch {
            expected: vec![cols],
            got: vec![in_curvature_diag.len()],
        });
    }
    for (index, &value) in out_curvature_diag.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(PgError::InvalidOp(format!(
                "out_curvature_diag[{index}] must be positive finite, got {value}"
            )));
        }
    }
    for (index, &value) in in_curvature_diag.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(PgError::InvalidOp(format!(
                "in_curvature_diag[{index}] must be positive finite, got {value}"
            )));
        }
    }
    Ok(())
}

struct SvdComponent {
    sigma: f64,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn truncated_svd_components(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Vec<SvdComponent> {
    if rows >= cols {
        let mut gram = vec![0.0f64; cols * cols];
        for i in 0..cols {
            for j in i..cols {
                let mut sum = 0.0f64;
                for row in 0..rows {
                    sum += matrix[row * cols + i] * matrix[row * cols + j];
                }
                gram[i * cols + j] = sum;
                gram[j * cols + i] = sum;
            }
        }
        jacobi_symmetric_eigen(gram, cols)
            .into_iter()
            .take(rank)
            .filter_map(|(lambda, v)| component_from_right_singular(matrix, rows, cols, lambda, v))
            .collect()
    } else {
        let mut gram = vec![0.0f64; rows * rows];
        for i in 0..rows {
            for j in i..rows {
                let mut sum = 0.0f64;
                for col in 0..cols {
                    sum += matrix[i * cols + col] * matrix[j * cols + col];
                }
                gram[i * rows + j] = sum;
                gram[j * rows + i] = sum;
            }
        }
        jacobi_symmetric_eigen(gram, rows)
            .into_iter()
            .take(rank)
            .filter_map(|(lambda, u)| component_from_left_singular(matrix, rows, cols, lambda, u))
            .collect()
    }
}

fn component_from_right_singular(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    lambda: f64,
    v: Vec<f64>,
) -> Option<SvdComponent> {
    if lambda <= 1e-24 {
        return None;
    }
    let sigma = lambda.max(0.0).sqrt();
    let mut u = vec![0.0f64; rows];
    for row in 0..rows {
        let mut sum = 0.0f64;
        for col in 0..cols {
            sum += matrix[row * cols + col] * v[col];
        }
        u[row] = sum / sigma;
    }
    let mut v = v;
    normalize_sign(&mut u, &mut v);
    Some(SvdComponent { sigma, u, v })
}

fn component_from_left_singular(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    lambda: f64,
    u: Vec<f64>,
) -> Option<SvdComponent> {
    if lambda <= 1e-24 {
        return None;
    }
    let sigma = lambda.max(0.0).sqrt();
    let mut v = vec![0.0f64; cols];
    for col in 0..cols {
        let mut sum = 0.0f64;
        for row in 0..rows {
            sum += matrix[row * cols + col] * u[row];
        }
        v[col] = sum / sigma;
    }
    let mut u = u;
    normalize_sign(&mut u, &mut v);
    Some(SvdComponent { sigma, u, v })
}

fn normalize_sign(u: &mut [f64], v: &mut [f64]) {
    let Some((_, &pivot)) = u
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
    else {
        return;
    };
    if pivot < 0.0 {
        for value in u {
            *value = -*value;
        }
        for value in v {
            *value = -*value;
        }
    }
}

fn jacobi_symmetric_eigen(mut a: Vec<f64>, n: usize) -> Vec<(f64, Vec<f64>)> {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n == 0 {
        return Vec::new();
    }

    let max_iter = 64usize.saturating_mul(n).saturating_mul(n).max(1);
    for _ in 0..max_iter {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                let value = a[i * n + j].abs();
                if value > max_off {
                    max_off = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..n {
            if k != p && k != q {
                let akp = a[k * n + p];
                let akq = a[k * n + q];
                let new_kp = c * akp - s * akq;
                let new_kq = s * akp + c * akq;
                a[k * n + p] = new_kp;
                a[p * n + k] = new_kp;
                a[k * n + q] = new_kq;
                a[q * n + k] = new_kq;
            }
        }

        let new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        let new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        for k in 0..n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }

    let mut eigen = (0..n)
        .map(|i| {
            let vector = (0..n).map(|row| v[row * n + i]).collect::<Vec<_>>();
            (a[i * n + i].max(0.0), vector)
        })
        .collect::<Vec<_>>();
    eigen.sort_by(|a, b| b.0.total_cmp(&a.0));
    eigen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_low_rank_reconstructs_exact_rank_update() {
        let rows = 4;
        let cols = 5;
        let rank = 2;
        let out_curv: Vec<f32> = vec![0.5, 1.25, 2.0, 3.5];
        let in_curv: Vec<f32> = vec![0.75, 1.1, 1.7, 2.4, 3.0];
        let mut full = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let low_rank_value = (row as f32 + 1.0) * (col as f32 * 0.3 + 0.7)
                    + (row as f32 * 0.2 - 0.5) * (col as f32 + 1.0).cos();
                full[row * cols + col] =
                    low_rank_value / out_curv[row].sqrt() / in_curv[col].sqrt();
            }
        }

        let delta = deltagolf_weighted_low_rank_delta(&full, &out_curv, &in_curv, rows, cols, rank)
            .unwrap();
        let mut materialized = vec![0.0f32; rows * cols];
        deltagolf_materialize_delta(&delta, &mut materialized).unwrap();

        assert_eq!(delta.rank, rank);
        assert_close_slice(&materialized, &full, 5e-5, "rank exact weighted delta");
        assert!(
            deltagolf_weighted_error(&materialized, &full, &out_curv, &in_curv, rows, cols)
                .unwrap()
                < 1e-9
        );
    }

    #[test]
    fn identity_curvature_rank_one_keeps_top_diagonal_component() {
        let rows = 3;
        let cols = 3;
        let full = vec![5.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let curv = vec![1.0f32; 3];

        let delta = deltagolf_weighted_low_rank_delta(&full, &curv, &curv, rows, cols, 1).unwrap();
        let mut materialized = vec![0.0f32; rows * cols];
        deltagolf_materialize_delta(&delta, &mut materialized).unwrap();

        let expected = vec![5.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_close_slice(&materialized, &expected, 1e-5, "rank one identity delta");
        assert_close_slice(&delta.singular_values, &[5.0], 1e-5, "singular values");
    }

    #[test]
    fn weighted_error_decreases_with_rank() {
        let rows = 5;
        let cols = 4;
        let full = deterministic_matrix(rows, cols);
        let out_curv = vec![0.6, 0.9, 1.4, 2.1, 3.2];
        let in_curv = vec![0.7, 1.3, 1.9, 2.8];
        let zero = vec![0.0f32; rows * cols];
        let rank1 =
            deltagolf_weighted_low_rank_delta(&full, &out_curv, &in_curv, rows, cols, 1).unwrap();
        let rank3 =
            deltagolf_weighted_low_rank_delta(&full, &out_curv, &in_curv, rows, cols, 3).unwrap();
        let mut mat1 = vec![0.0f32; rows * cols];
        let mut mat3 = vec![0.0f32; rows * cols];
        deltagolf_materialize_delta(&rank1, &mut mat1).unwrap();
        deltagolf_materialize_delta(&rank3, &mut mat3).unwrap();

        let zero_err =
            deltagolf_weighted_error(&zero, &full, &out_curv, &in_curv, rows, cols).unwrap();
        let rank1_err =
            deltagolf_weighted_error(&mat1, &full, &out_curv, &in_curv, rows, cols).unwrap();
        let rank3_err =
            deltagolf_weighted_error(&mat3, &full, &out_curv, &in_curv, rows, cols).unwrap();
        assert!(rank1_err < zero_err);
        assert!(rank3_err < rank1_err);
    }

    #[test]
    fn weighted_low_rank_rejects_non_positive_curvature() {
        let full = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = deltagolf_weighted_low_rank_delta(&full, &[1.0, 0.0], &[1.0, 1.0], 2, 2, 1)
            .unwrap_err();
        assert!(err.to_string().contains("out_curvature_diag[1]"));
    }

    fn deterministic_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let x = i as f32 + 1.0;
                0.6 * (x * 0.17).sin() + 0.4 * (x * 0.31).cos() + 0.05 * (i % cols) as f32
            })
            .collect()
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
