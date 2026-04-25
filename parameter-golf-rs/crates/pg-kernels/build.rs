use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_cuda_cpp)");

    // Only attempt to build CUDA/C++ extensions if the 'cuda' feature is enabled.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    // Check if nvcc is available
    if Command::new("nvcc").arg("--version").status().is_err() {
        println!("cargo:warning=nvcc not found. Skipping CUDA/C++ F32 SDPA compilation.");
        return;
    }

    // We only compile the CUDA/C++ F32 SDPA backend if nvcc is found.
    // Ensure `cc` is listed in build-dependencies.
    let mut build = cc::Build::new();

    build
        .cuda(true)
        .flag("-O3")
        // Allow C++17 for CUDA/C++ attention sources.
        .flag("-std=c++17")
        .file("cpp/sdpa.cu")
        .compile("naive_sdpa_f32");

    println!("cargo:rustc-cfg=has_cuda_cpp");

    // Link against cudart. The current C++ SDPA backend does not use cuDNN.
    println!("cargo:rustc-link-lib=cudart");

    // Re-run if the C++ file changes
    println!("cargo:rerun-if-changed=cpp/sdpa.cu");
    println!("cargo:rerun-if-changed=build.rs");
}
