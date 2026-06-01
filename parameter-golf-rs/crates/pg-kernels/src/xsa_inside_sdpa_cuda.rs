use cudarc::driver::CudaStream;
use pg_core::error::{PgError, PgResult};
use std::ffi::c_void;
use std::sync::Arc;

unsafe extern "C" {
    fn run_naive_xsa_inside_sdpa_f32_forward(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
    ) -> i32;

    fn run_naive_xsa_inside_sdpa_f32_backward(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        batch: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
    ) -> i32;
}

/// Correctness-first CUDA F32 reference for fused causal SDPA + XSA.
///
/// This is an opt-in extension boundary only. It is compiled with the same
/// CUDA/nvcc gate as the existing C++ SDPA backend and is not wired into the
/// model runtime.
pub struct CudaXsaInsideSdpa {
    stream: Arc<CudaStream>,
}

#[derive(Clone, Copy)]
struct CheckedDims {
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
}

impl CudaXsaInsideSdpa {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }

    pub fn is_available() -> bool {
        true
    }

    pub fn backend_name(&self) -> &'static str {
        "naive_f32_cuda_xsa_inside_sdpa"
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        softmax_scale: f32,
    ) -> PgResult<()> {
        validate_ptrs(&[q, k, v, out])?;
        let dims = validate_dims(batch, seq_len, num_heads, num_kv_heads, head_dim)?;

        unsafe {
            let status = run_naive_xsa_inside_sdpa_f32_forward(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                dims.batch,
                dims.seq_len,
                dims.num_heads,
                dims.num_kv_heads,
                dims.head_dim,
                softmax_scale,
            );
            cuda_status(status, "XSA-inside-SDPA forward")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        softmax_scale: f32,
    ) -> PgResult<()> {
        validate_ptrs(&[q, k, v, grad_out, grad_q, grad_k, grad_v])?;
        let dims = validate_dims(batch, seq_len, num_heads, num_kv_heads, head_dim)?;

        unsafe {
            let status = run_naive_xsa_inside_sdpa_f32_backward(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                dims.batch,
                dims.seq_len,
                dims.num_heads,
                dims.num_kv_heads,
                dims.head_dim,
                softmax_scale,
            );
            cuda_status(status, "XSA-inside-SDPA backward")
        }
    }
}

fn validate_ptrs(ptrs: &[u64]) -> PgResult<()> {
    if ptrs.iter().any(|&ptr| ptr == 0) {
        return Err(PgError::InvalidOp(
            "XSA-inside-SDPA CUDA backend received a null device pointer".into(),
        ));
    }
    Ok(())
}

fn validate_dims(
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> PgResult<CheckedDims> {
    if batch == 0 || seq_len == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(PgError::InvalidOp(
            "XSA-inside-SDPA CUDA backend requires non-zero dimensions".into(),
        ));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(PgError::InvalidOp(
            "XSA-inside-SDPA CUDA backend requires num_heads divisible by num_kv_heads".into(),
        ));
    }

    Ok(CheckedDims {
        batch: checked_i32(batch, "batch")?,
        seq_len: checked_i32(seq_len, "seq_len")?,
        num_heads: checked_i32(num_heads, "num_heads")?,
        num_kv_heads: checked_i32(num_kv_heads, "num_kv_heads")?,
        head_dim: checked_i32(head_dim, "head_dim")?,
    })
}

fn checked_i32(value: usize, name: &str) -> PgResult<i32> {
    i32::try_from(value).map_err(|_| {
        PgError::InvalidOp(format!(
            "XSA-inside-SDPA CUDA dimension {name} exceeds i32 range: {value}"
        ))
    })
}

fn cuda_status(status: i32, op: &str) -> PgResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(PgError::InvalidOp(format!(
            "{op} failed with status code {status}"
        )))
    }
}
