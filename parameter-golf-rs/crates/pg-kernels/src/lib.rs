pub mod activations;
pub mod attention;
pub mod bigram_hash;
pub mod complementary;
pub mod cross_entropy;
pub mod fusion;
pub mod linear;
pub mod rms_norm;
pub mod rope;
pub mod smear_gate;
pub mod xsa;

pub const COMPILED_CUDA_ARCHES: Option<&str> = option_env!("PG_COMPILED_CUDA_ARCHES");
pub const COMPILED_ARCH_PROFILE: Option<&str> = option_env!("PG_COMPILED_ARCH_PROFILE");

#[cfg(feature = "cuda")]
pub mod gemm;

#[cfg(feature = "cuda")]
pub mod gpu_kernels;

#[cfg(feature = "cuda")]
pub mod flash_attn;

#[cfg(feature = "cuda")]
pub mod output_ce;

#[cfg(all(feature = "cuda", has_cuda_cpp))]
pub mod xsa_inside_sdpa_cuda;
