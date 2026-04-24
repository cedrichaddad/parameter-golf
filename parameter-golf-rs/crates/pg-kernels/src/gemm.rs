use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t,
};
use cudarc::driver::CudaStream;

use pg_core::error::{PgError, PgResult};

pub struct GemmEngine {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
}

impl GemmEngine {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        let blas = CudaBlas::new(stream.clone()).map_err(|e| PgError::CuBlas(e.to_string()))?;
        Ok(Self { blas, stream })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub unsafe fn matmul_f32(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        // Row-major linear projection:
        //   a = X [m, k]
        //   b = W [n, k]
        //   c = Y [m, n]
        // with CPU semantics Y = X @ W^T.
        //
        // cuBLAS is column-major, so we compute:
        //   Y^T [n, m] = W [n, k] @ X^T [k, m]
        // by interpreting the row-major buffers as transposed column-major views.
        unsafe { self.matmul_f32_bt(a, b, c, m, n, k, alpha, beta) }
    }

    pub unsafe fn batched_matmul_f32(
        &self,
        a: u64,
        b: u64,
        c: u64,
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

        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_b,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_a,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_strided_batched_ex failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_f32_bt(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_ex failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_f32_nn(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        // Row-major:
        //   A [m, k], B [k, n], C [m, n]
        //   C = A @ B
        //
        // cuBLAS column-major view:
        //   C^T [n, m] = B^T [n, k] @ A^T [k, m]
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_ex nn failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_f32_tn(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        // Row-major:
        //   A [k, m], B [k, n], C [m, n]
        //   C = A^T @ B
        //
        // cuBLAS column-major view:
        //   C^T [n, m] = B^T [n, k] @ A [k, m]
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_T,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                m as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_ex tn failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn linear_backward_input_f32(
        &self,
        dy: u64,
        w: u64,
        dx: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        // Forward uses y[t,out] = x[t,in] @ w[out,in]^T.
        // Backward input is dx[t,in] += dy[t,out] @ w[out,in].
        unsafe { self.matmul_f32_nn(dy, w, dx, tokens, in_dim, out_dim, alpha, beta) }
    }

    pub unsafe fn linear_backward_weight_f32(
        &self,
        dy: u64,
        x: u64,
        dw: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        // Forward uses y[t,out] = x[t,in] @ w[out,in]^T.
        // Backward weight is dw[out,in] += dy[t,out]^T @ x[t,in].
        unsafe { self.matmul_f32_tn(dy, x, dw, out_dim, in_dim, tokens, alpha, beta) }
    }
}
