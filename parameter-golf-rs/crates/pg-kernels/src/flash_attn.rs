use cudarc::driver::CudaStream;
use pg_core::error::{PgError, PgResult};
use std::sync::Arc;

#[cfg(has_cuda_cpp)]
use std::ffi::c_void;

#[cfg(has_cuda_cpp)]
unsafe extern "C" {
    fn run_cudnn_sdpa_f32(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch_tokens: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;
}

/// CUDA attention backend compiled from `cpp/sdpa.cu`.
pub struct FlashAttention {
    #[cfg_attr(not(has_cuda_cpp), allow(dead_code))]
    stream: Arc<CudaStream>,
}

impl FlashAttention {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        #[cfg(has_cuda_cpp)]
        {
            Ok(Self { stream })
        }
        #[cfg(not(has_cuda_cpp))]
        {
            let _ = stream;
            Err(PgError::InvalidOp(
                "CUDA C++ attention backend was not compiled for this build".into(),
            ))
        }
    }

    pub fn is_available() -> bool {
        cfg!(has_cuda_cpp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cuda_cpp)]
        unsafe {
            let status = run_cudnn_sdpa_f32(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                batch_tokens as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "CUDA attention backend failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cuda_cpp))]
        {
            let _ = (
                q,
                k,
                v,
                out,
                batch_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "CUDA C++ attention backend was not compiled for this build".into(),
            ))
        }
    }
}
