use std::sync::Arc;

use cudarc::cublas::{CudaBlas, GemmConfig, Gemm, StridedBatchedConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaStream, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use pg_core::error::{PgError, PgResult};

/// cuBLAS GEMM wrapper for the competition's critical matrix shapes.
///
/// All GEMM in the model (row-major, bf16 with f32 accumulation):
/// - Attention Q/K/V projection: [B*T, 512] x [512, 512] or [512, 256]
/// - Attention output:           [B*T, 512] x [512, 512]
/// - MLP up:                     [B*T, 512] x [1536, 512]^T
/// - MLP down:                   [B*T, 1536] x [512, 1536]^T
///
/// Newton-Schulz uses strided batched GEMM:
/// - X @ X^T:  [B, M, N] x [B, N, M] -> [B, M, M]
/// - coeff @ X: [B, M, M] x [B, M, N] -> [B, M, N]
pub struct GemmEngine {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
}

impl GemmEngine {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| PgError::CuBlas(e.to_string()))?;
        Ok(Self { blas, stream })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// C = alpha * A @ B + beta * C  (row-major bf16, f32 accumulation)
    ///
    /// A: [m, k] row-major, B: [k, n] row-major, C: [m, n] row-major.
    ///
    /// cuBLAS is column-major, so we compute C^T = B^T @ A^T which is
    /// equivalent to passing B as the first matrix with OP_N and A as the
    /// second with OP_N, with lda=n, ldb=k, ldc=n.
    ///
    /// # Safety
    /// Caller must ensure a has m*k elements, b has k*n, c has m*n.
    pub unsafe fn matmul_bf16(
        &self,
        a: &CudaSlice<bf16>,
        b: &CudaSlice<bf16>,
        c: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: bf16::from_f32(alpha),
            lda: n as i32,
            ldb: k as i32,
            beta: bf16::from_f32(beta),
            ldc: n as i32,
        };
        unsafe {
            self.blas.gemm(cfg, b, a, c)
                .map_err(|e| PgError::CuBlas(e.to_string()))?;
        }
        Ok(())
    }

    /// Strided batched GEMM for Newton-Schulz orthogonalization.
    ///
    /// A: [batch, m, k], B: [batch, k, n], C: [batch, m, n].
    /// All row-major bf16 with f32 accumulation.
    ///
    /// # Safety
    /// Caller must ensure correct tensor sizes and strides.
    pub unsafe fn batched_matmul_bf16(
        &self,
        a: &CudaSlice<bf16>,
        b: &CudaSlice<bf16>,
        c: &mut CudaSlice<bf16>,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        let stride_a = (m * k) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: bf16::from_f32(alpha),
                lda: n as i32,
                ldb: k as i32,
                beta: bf16::from_f32(beta),
                ldc: n as i32,
            },
            batch_size: batch as i32,
            stride_a: stride_b, // cuBLAS swap: B is first arg
            stride_b: stride_a, // cuBLAS swap: A is second arg
            stride_c,
        };
        unsafe {
            self.blas.gemm_strided_batched(cfg, b, a, c)
                .map_err(|e| PgError::CuBlas(e.to_string()))?;
        }
        Ok(())
    }

    /// Transposed B variant: C = alpha * A @ B^T + beta * C
    ///
    /// A: [m, k], B: [n, k] (stored as [n, k] row-major, transposed to [k, n]),
    /// C: [m, n].
    ///
    /// In cuBLAS column-major land: C^T = B^T_T @ A^T = B @ A^T
    /// so transa = OP_T, transb = OP_N
    ///
    /// # Safety
    /// Caller must ensure a has m*k elements, b has n*k, c has m*n.
    pub unsafe fn matmul_bf16_bt(
        &self,
        a: &CudaSlice<bf16>,
        b: &CudaSlice<bf16>,
        c: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: bf16::from_f32(alpha),
            lda: k as i32,
            ldb: k as i32,
            beta: bf16::from_f32(beta),
            ldc: n as i32,
        };
        unsafe {
            self.blas.gemm(cfg, b, a, c)
                .map_err(|e| PgError::CuBlas(e.to_string()))?;
        }
        Ok(())
    }
}
