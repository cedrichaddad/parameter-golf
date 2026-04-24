use std::path::PathBuf;

use pg_model::{QuantScheme, RunMode, RunSpec, TrainBackend, VariantFamily};
use pg_train::VariantRunner;

fn main() {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        std::process::exit(2);
    };

    match command.as_str() {
        "run" => {
            let mut spec_path: Option<PathBuf> = None;
            let mut builtin: Option<VariantFamily> = None;
            let mut mode: Option<RunMode> = None;
            let mut backend: Option<TrainBackend> = None;
            let mut train_data_pattern: Option<String> = None;
            let mut validation_data_pattern: Option<String> = None;
            let mut artifact_path: Option<String> = None;
            let mut rank: Option<usize> = None;
            let mut world_size: Option<usize> = None;
            let mut batch_tokens: Option<usize> = None;
            let mut seq_len: Option<usize> = None;
            let mut total_iterations: Option<usize> = None;
            let mut max_wallclock_seconds: Option<f32> = None;
            let mut tokenizer_vocab_path: Option<String> = None;
            let mut eval_max_tokens: Option<usize> = None;
            let mut eval_stride: Option<usize> = None;
            let mut quant_scheme: Option<QuantScheme> = None;
            let mut prune_keep_ratio: Option<f32> = None;
            let mut fast_bank_updates = false;
            let mut allow_unsupported_variants = false;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--spec" => spec_path = args.next().map(PathBuf::from),
                    "--builtin" => {
                        builtin = args.next().as_deref().and_then(parse_family);
                    }
                    "--mode" => {
                        mode = args.next().as_deref().and_then(parse_mode);
                    }
                    "--backend" => {
                        backend = args.next().as_deref().and_then(parse_backend);
                    }
                    "--train-data" => train_data_pattern = args.next(),
                    "--val-data" => validation_data_pattern = args.next(),
                    "--artifact" => artifact_path = args.next(),
                    "--rank" => rank = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--world-size" => {
                        world_size = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--batch-tokens" => {
                        batch_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--seq-len" => seq_len = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--total-iterations" => {
                        total_iterations = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--max-wallclock-seconds" => {
                        max_wallclock_seconds = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--tokenizer-vocab" => tokenizer_vocab_path = args.next(),
                    "--eval-max-tokens" => {
                        eval_max_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--eval-stride" => {
                        eval_stride = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--quant-scheme" => {
                        quant_scheme = args.next().as_deref().and_then(parse_quant_scheme)
                    }
                    "--prune-keep-ratio" => {
                        prune_keep_ratio = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--fast-bank-updates" => fast_bank_updates = true,
                    "--allow-unsupported-variants" => allow_unsupported_variants = true,
                    _ => {}
                }
            }

            let mut run_spec = if let Some(path) = spec_path {
                RunSpec::load(&path).expect("failed to load spec")
            } else {
                RunSpec::for_family(builtin.unwrap_or(VariantFamily::BaselineSp8192))
            };
            if let Some(mode) = mode {
                run_spec.mode = mode;
            }
            if let Some(backend) = backend {
                run_spec.train.backend = backend;
            }
            if let Some(pattern) = train_data_pattern {
                run_spec.train.train_data_pattern = Some(pattern);
            }
            if let Some(pattern) = validation_data_pattern {
                run_spec.train.validation_data_pattern = Some(pattern);
            }
            if let Some(path) = artifact_path {
                run_spec.train.artifact_path = path;
            }
            if let Some(value) = rank {
                run_spec.train.rank = value;
            }
            if let Some(value) = world_size {
                run_spec.train.world_size = value;
            }
            if let Some(value) = batch_tokens {
                run_spec.train.batch_tokens = value;
            }
            if let Some(value) = seq_len {
                run_spec.train.seq_len = value;
                run_spec.model.train_seq_len = value.max(run_spec.model.train_seq_len);
            }
            if let Some(value) = total_iterations {
                run_spec.train.total_iterations = value;
            }
            if let Some(value) = max_wallclock_seconds {
                run_spec.train.max_wallclock_seconds = value;
            }
            if let Some(path) = tokenizer_vocab_path {
                run_spec.eval.tokenizer_vocab_path = Some(path);
            }
            if let Some(value) = eval_max_tokens {
                run_spec.eval.max_tokens = Some(value);
            }
            if let Some(value) = eval_stride {
                run_spec.eval.stride = value;
            }
            if let Some(value) = quant_scheme {
                run_spec.quant.scheme = value;
            }
            if let Some(value) = prune_keep_ratio {
                run_spec.quant.prune_keep_ratio = Some(value);
            }
            if fast_bank_updates {
                run_spec.train.fast_bank_updates = true;
            }
            if allow_unsupported_variants {
                run_spec.allow_unsupported_variants = true;
            }
            let result = match VariantRunner::new(run_spec.clone())
                .and_then(|runner| runner.run(run_spec.mode))
            {
                Ok(result) => result,
                Err(err) => {
                    eprintln!("variant run failed: {err}");
                    std::process::exit(1);
                }
            };
            println!("run_name={}", result.run_name);
            println!("mode={:?}", result.mode);
            println!("train_backend={:?}", result.train_backend);
            println!("variant_fingerprint={}", result.variant_fingerprint);
            println!("steps_completed={}", result.steps_completed);
            println!("train_loss={:.6}", result.train_loss);
            println!("ms_per_step={:.3}", result.ms_per_step);
            println!("wallclock_seconds={:.3}", result.wallclock_seconds);
            println!("rank={}", result.rank);
            println!("world_size={}", result.world_size);
            println!("distributed_sync={}", result.distributed_sync);
            println!("bank_update_backend={}", result.bank_update_backend);
            println!("train_data_source={}", result.train_data_source);
            println!("bpb_byte_source={}", result.bpb_byte_source);
            if let Some(bytes) = result.artifact_bytes {
                println!("artifact_bytes={bytes}");
            }
            if let Some(ok) = result.artifact_budget_ok {
                println!("artifact_budget_ok={ok}");
            }
            if let Some(bpb) = result.proxy_bpb {
                println!("proxy_bpb={bpb:.6}");
            }
            if let Some(source) = result.proxy_metric_source {
                println!("proxy_metric_source={source}");
            }
            if let Some(tokens) = result.eval_tokens {
                println!("eval_tokens={tokens}");
            }
            if let Some(loss) = result.eval_loss {
                println!("eval_loss={loss:.6}");
            }
            if let Some(bpb) = result.final_bpb {
                println!("final_bpb={bpb:.6}");
            }
        }
        "sweep" => {
            let mut mode = RunMode::Smoke;
            let mut backend: Option<TrainBackend> = None;
            let mut train_data_pattern: Option<String> = None;
            let mut validation_data_pattern: Option<String> = None;
            let mut rank: Option<usize> = None;
            let mut world_size: Option<usize> = None;
            let mut batch_tokens: Option<usize> = None;
            let mut seq_len: Option<usize> = None;
            let mut total_iterations: Option<usize> = None;
            let mut max_wallclock_seconds: Option<f32> = None;
            let mut tokenizer_vocab_path: Option<String> = None;
            let mut eval_max_tokens: Option<usize> = None;
            let mut eval_stride: Option<usize> = None;
            let mut quant_scheme: Option<QuantScheme> = None;
            let mut prune_keep_ratio: Option<f32> = None;
            let mut fast_bank_updates = false;
            let mut allow_unsupported_variants = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--mode" => {
                        if let Some(raw) = args.next() {
                            mode = parse_mode(&raw).unwrap_or(mode);
                        }
                    }
                    "--backend" => backend = args.next().as_deref().and_then(parse_backend),
                    "--train-data" => train_data_pattern = args.next(),
                    "--val-data" => validation_data_pattern = args.next(),
                    "--rank" => rank = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--world-size" => {
                        world_size = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--batch-tokens" => {
                        batch_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--seq-len" => seq_len = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--total-iterations" => {
                        total_iterations = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--max-wallclock-seconds" => {
                        max_wallclock_seconds = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--tokenizer-vocab" => tokenizer_vocab_path = args.next(),
                    "--eval-max-tokens" => {
                        eval_max_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--eval-stride" => {
                        eval_stride = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--quant-scheme" => {
                        quant_scheme = args.next().as_deref().and_then(parse_quant_scheme)
                    }
                    "--prune-keep-ratio" => {
                        prune_keep_ratio = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--fast-bank-updates" => fast_bank_updates = true,
                    "--allow-unsupported-variants" => allow_unsupported_variants = true,
                    _ => {}
                }
            }
            let default_families = [VariantFamily::BaselineSp8192, VariantFamily::XsaAllSp8192];
            let all_families = [
                VariantFamily::BaselineSp8192,
                VariantFamily::XsaAllSp8192,
                VariantFamily::RecurrenceMidSp8192,
                VariantFamily::ParallelResidSp8192,
                VariantFamily::HybridCompetitiveSp8192,
            ];
            let families: &[VariantFamily] = if allow_unsupported_variants {
                &all_families
            } else {
                &default_families
            };
            for &family in families {
                let mut run_spec = RunSpec::for_family(family);
                run_spec.mode = mode;
                if let Some(backend) = backend {
                    run_spec.train.backend = backend;
                }
                if allow_unsupported_variants {
                    run_spec.allow_unsupported_variants = true;
                }
                run_spec.train.artifact_path =
                    format!("/tmp/pg_{family:?}_{mode:?}.pgrs").to_lowercase();
                if let Some(pattern) = train_data_pattern.clone() {
                    run_spec.train.train_data_pattern = Some(pattern);
                }
                if let Some(pattern) = validation_data_pattern.clone() {
                    run_spec.train.validation_data_pattern = Some(pattern);
                }
                if let Some(value) = rank {
                    run_spec.train.rank = value;
                }
                if let Some(value) = world_size {
                    run_spec.train.world_size = value;
                }
                if let Some(value) = batch_tokens {
                    run_spec.train.batch_tokens = value;
                }
                if let Some(value) = seq_len {
                    run_spec.train.seq_len = value;
                    run_spec.model.train_seq_len = value.max(run_spec.model.train_seq_len);
                }
                if let Some(value) = total_iterations {
                    run_spec.train.total_iterations = value;
                }
                if let Some(value) = max_wallclock_seconds {
                    run_spec.train.max_wallclock_seconds = value;
                }
                if let Some(path) = tokenizer_vocab_path.clone() {
                    run_spec.eval.tokenizer_vocab_path = Some(path);
                }
                if let Some(value) = eval_max_tokens {
                    run_spec.eval.max_tokens = Some(value);
                }
                if let Some(value) = eval_stride {
                    run_spec.eval.stride = value;
                }
                if let Some(value) = quant_scheme {
                    run_spec.quant.scheme = value;
                }
                if let Some(value) = prune_keep_ratio {
                    run_spec.quant.prune_keep_ratio = Some(value);
                }
                if fast_bank_updates {
                    run_spec.train.fast_bank_updates = true;
                }
                match VariantRunner::new(run_spec.clone()).and_then(|runner| runner.run(mode)) {
                    Ok(result) => {
                        println!(
                            "variant={:?} status=ok backend={:?} fingerprint={} steps={} loss={:.6} ms_per_step={:.3} rank={} world_size={} distributed_sync={} bank_update_backend={} train_data_source={} bpb_byte_source={} proxy_bpb={} proxy_metric_source={} final_bpb={} artifact_budget_ok={}",
                            family,
                            result.train_backend,
                            result.variant_fingerprint,
                            result.steps_completed,
                            result.train_loss,
                            result.ms_per_step,
                            result.rank,
                            result.world_size,
                            result.distributed_sync,
                            result.bank_update_backend,
                            result.train_data_source,
                            result.bpb_byte_source,
                            result
                                .proxy_bpb
                                .map(|v| format!("{v:.6}"))
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .proxy_metric_source
                                .clone()
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .final_bpb
                                .map(|v| format!("{v:.6}"))
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .artifact_budget_ok
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "unknown".to_string()),
                        );
                    }
                    Err(err) => {
                        println!("variant={:?} status=skipped reason={}", family, err);
                    }
                }
            }
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!(
        "  pg-train run [--spec spec.toml] [--builtin baseline_sp8192] [--mode smoke|proxy|record]"
    );
    eprintln!("               [--backend cpu|cuda-single|cuda-single-parity|cuda-distributed]");
    eprintln!("               [--train-data glob] [--val-data glob] [--artifact path]");
    eprintln!("               [--rank n] [--world-size n] [--batch-tokens n] [--seq-len n]");
    eprintln!("               [--total-iterations n] [--max-wallclock-seconds n]");
    eprintln!("               [--tokenizer-vocab path] [--eval-max-tokens n] [--eval-stride n]");
    eprintln!("               [--quant-scheme gptq_lite_int6|mixed_int5_int6|aggressive|tight_int7_int4]");
    eprintln!("               [--prune-keep-ratio f]");
    eprintln!("               [--fast-bank-updates] [--allow-unsupported-variants]");
    eprintln!("  pg-train sweep [--mode smoke|proxy]");
    eprintln!("                 [--backend cpu|cuda-single|cuda-single-parity|cuda-distributed]");
    eprintln!("                 [--train-data glob] [--val-data glob]");
    eprintln!("                 [--rank n] [--world-size n] [--batch-tokens n] [--seq-len n]");
    eprintln!("                 [--total-iterations n] [--max-wallclock-seconds n]");
    eprintln!("                 [--tokenizer-vocab path] [--eval-max-tokens n] [--eval-stride n]");
    eprintln!("                 [--quant-scheme gptq_lite_int6|mixed_int5_int6|aggressive|tight_int7_int4]");
    eprintln!("                 [--prune-keep-ratio f]");
    eprintln!("                 [--fast-bank-updates] [--allow-unsupported-variants]");
}

fn parse_mode(raw: &str) -> Option<RunMode> {
    match raw {
        "smoke" => Some(RunMode::Smoke),
        "proxy" => Some(RunMode::Proxy),
        "record" => Some(RunMode::Record),
        _ => None,
    }
}

fn parse_backend(raw: &str) -> Option<TrainBackend> {
    match raw {
        "cpu" => Some(TrainBackend::Cpu),
        "cuda-single" => Some(TrainBackend::CudaSingle),
        "cuda-single-parity" => Some(TrainBackend::CudaSingleParity),
        "cuda-distributed" => Some(TrainBackend::CudaDistributed),
        _ => None,
    }
}

fn parse_quant_scheme(raw: &str) -> Option<QuantScheme> {
    match raw {
        "none" => Some(QuantScheme::None),
        "gptq_lite_int6" => Some(QuantScheme::GptqLiteInt6),
        "mixed_int5_int6" => Some(QuantScheme::MixedInt5Int6),
        "aggressive" => Some(QuantScheme::Aggressive),
        "tight_int7_int4" => Some(QuantScheme::TightInt7Int4),
        _ => None,
    }
}

fn parse_family(raw: &str) -> Option<VariantFamily> {
    match raw {
        "baseline_sp8192" => Some(VariantFamily::BaselineSp8192),
        "xsa_all_sp8192" => Some(VariantFamily::XsaAllSp8192),
        "recurrence_mid_sp8192" => Some(VariantFamily::RecurrenceMidSp8192),
        "parallel_resid_sp8192" => Some(VariantFamily::ParallelResidSp8192),
        "hybrid_competitive_sp8192" => Some(VariantFamily::HybridCompetitiveSp8192),
        _ => None,
    }
}
