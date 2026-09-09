use std::path::PathBuf;

use pg_core::{PgError, PgResult};
use pg_local::{
    ArtifactLabOptions, BackendCheckOptions, DistSimOptions, LiteBackendKind, LiteBenchmarkOptions,
    LiteInitOptions, LiteRunOptions, LiteSuiteOptions, LiteVerifyArtifactOptions, ModelGolfOptions,
    ModelGolfSection, ProofBundleOptions, TraceIndexOptions, VerifyOptions,
    WindTunnelCalibrationOptions, WindTunnelCorpusOptions, WindTunnelOptions,
    WindTunnelReviewerOptions, WindTunnelScenarioOptions, WindTunnelSuiteOptions, run_artifact_lab,
    run_backend_check, run_dist_sim, run_lite, run_lite_benchmark, run_lite_init, run_lite_suite,
    run_lite_verify_artifact, run_modelgolf_plan, run_proof_bundle, run_trace_index, run_verify,
    run_wind_tunnel, run_wind_tunnel_calibrate, run_wind_tunnel_corpus, run_wind_tunnel_report,
    run_wind_tunnel_scenario, run_wind_tunnel_suite,
};

fn main() {
    if let Err(err) = real_main() {
        eprintln!("pg-local error: {err}");
        std::process::exit(1);
    }
}

fn real_main() -> PgResult<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_help();
        return Ok(());
    }
    let command = args.remove(0);
    if command == "help" {
        if args.is_empty() {
            print_help();
        } else {
            print_command_help(&args.join(" "));
        }
        return Ok(());
    }
    if args.first().map(String::as_str).is_some_and(is_help_arg) {
        print_command_help(&command);
        return Ok(());
    }
    match command.as_str() {
        "verify" => {
            let options = VerifyOptions {
                spec: required_path(&mut args, "--spec")?,
                artifact: optional_path(&mut args, "--artifact")?,
                artifact_audit: optional_path(&mut args, "--artifact-audit")?,
                score_first_log: optional_path(&mut args, "--score-first-log")?,
                output: optional_path(&mut args, "--output")?,
            };
            run_verify(options)?;
        }
        "proof-bundle" => {
            let options = ProofBundleOptions {
                spec: required_path(&mut args, "--spec")?,
                artifact: optional_path(&mut args, "--artifact")?,
                trace: optional_path(&mut args, "--trace")?,
                trace_dir: optional_path(&mut args, "--trace-dir")?,
                lite_config: required_path(&mut args, "--lite-config")?,
                lite_config_dir: optional_path(&mut args, "--lite-config-dir")?,
                output_dir: required_path(&mut args, "--output-dir")?,
            };
            run_proof_bundle(options)?;
        }
        "dist-sim" => {
            let options = DistSimOptions {
                spec: required_path(&mut args, "--spec")?,
                world_size: optional_usize(&mut args, "--world-size")?.unwrap_or(8),
                steps: optional_usize(&mut args, "--steps")?.unwrap_or(3),
                seed: optional_u64(&mut args, "--seed")?.unwrap_or(1),
                output: optional_path(&mut args, "--output")?,
            };
            run_dist_sim(options)?;
        }
        "artifact-lab" => {
            let options = ArtifactLabOptions {
                spec: required_path(&mut args, "--spec")?,
                artifact: optional_path(&mut args, "--artifact")?,
                sweep: optional_string(&mut args, "--sweep")?
                    .unwrap_or_else(|| "quant,lqer,compression".to_string())
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().to_string())
                    .collect(),
                mini_train: optional_path(&mut args, "--mini-train")?,
                mini_val: optional_path(&mut args, "--mini-val")?,
                output: optional_path(&mut args, "--output")?,
            };
            run_artifact_lab(options)?;
        }
        "modelgolf" => {
            if args.first().map(String::as_str).is_some_and(is_help_arg) {
                print_command_help("modelgolf");
                return Ok(());
            }
            let section = parse_modelgolf_section(&mut args)?;
            if args.first().map(String::as_str).is_some_and(is_help_arg) {
                print_command_help("modelgolf");
                return Ok(());
            }
            let options = ModelGolfOptions {
                section,
                spec: required_path(&mut args, "--spec")?,
                output: optional_path(&mut args, "--output")?,
                hardware: optional_string(&mut args, "--hardware")?
                    .unwrap_or_else(|| "local_planner".to_string()),
                runtime: optional_string(&mut args, "--runtime")?
                    .unwrap_or_else(|| "pg-local".to_string()),
                memory_budget_bytes: optional_usize(&mut args, "--memory-budget-bytes")?,
                latency_target_ms: optional_f64(&mut args, "--latency-target-ms")?,
                quality_budget_ppl_pct: optional_f64(&mut args, "--quality-budget-ppl-pct")?,
                context: optional_usize(&mut args, "--context")?,
                batch: optional_usize(&mut args, "--batch")?,
                artifact_budget_bytes: optional_usize(&mut args, "--artifact-budget-bytes")?,
                delta_budget_bytes: optional_usize(&mut args, "--delta-budget-bytes")?,
                proof_artifact: optional_path(&mut args, "--proof-artifact")?,
                quality_calibration: optional_path(&mut args, "--quality-calibration")?,
                release_evidence: optional_path(&mut args, "--release-evidence")?,
                release_artifact: optional_path(&mut args, "--release-artifact")?,
                validation_dataset_id: optional_string(&mut args, "--validation-dataset-id")?,
                validation_command: optional_string(&mut args, "--validation-command")?,
                heldout_bpb: optional_f64(&mut args, "--heldout-bpb")?,
                baseline_bpb: optional_f64(&mut args, "--baseline-bpb")?,
                decode_tokens_per_second: optional_f64(&mut args, "--decode-tokens-per-second")?,
                backend_id: optional_string(&mut args, "--backend-id")?,
                kernel_id: optional_string(&mut args, "--kernel-id")?,
                long_context_dataset_id: optional_string(&mut args, "--long-context-dataset-id")?,
                fused_runtime: optional_bool(&mut args, "--fused-runtime")?,
                parity_pass: optional_bool(&mut args, "--parity-pass")?,
                long_context_bpb_delta_pct: optional_f64(
                    &mut args,
                    "--long-context-bpb-delta-pct",
                )?,
                speedup_x: optional_f64(&mut args, "--speedup-x")?,
                calibration_dataset_id: optional_string(&mut args, "--calibration-dataset-id")?,
                production_svd_validated: optional_bool(&mut args, "--production-svd-validated")?,
                calibrated_tensor_sensitivity: optional_bool(
                    &mut args,
                    "--calibrated-tensor-sensitivity",
                )?,
                equal_byte_bpb_delta: optional_f64(&mut args, "--equal-byte-bpb-delta")?,
                domain_dataset_id: optional_string(&mut args, "--domain-dataset-id")?,
                trained_delta_bytes: optional_usize(&mut args, "--trained-delta-bytes")?,
                legality_pass: optional_bool(&mut args, "--legality-pass")?,
                score_first_trace_or_review: optional_bool(
                    &mut args,
                    "--score-first-trace-or-review",
                )?,
                equal_byte_domain_bpb_delta: optional_f64(
                    &mut args,
                    "--equal-byte-domain-bpb-delta",
                )?,
                training_run_id: optional_string(&mut args, "--training-run-id")?,
                gpu_backend_integrated: optional_bool(&mut args, "--gpu-backend-integrated")?,
                proxy_calibrated: optional_bool(&mut args, "--proxy-calibrated")?,
                post_export_bpb_delta_vs_posthoc: optional_f64(
                    &mut args,
                    "--post-export-bpb-delta-vs-posthoc",
                )?,
                measurement_run_id: optional_string(&mut args, "--measurement-run-id")?,
                power_meter_id: optional_string(&mut args, "--power-meter-id")?,
                wall_time_seconds: optional_f64(&mut args, "--wall-time-seconds")?,
                average_power_watts: optional_f64(&mut args, "--average-power-watts")?,
                energy_joules: optional_f64(&mut args, "--energy-joules")?,
                telemetry_validated: optional_bool(&mut args, "--telemetry-validated")?,
                distributed_backend_id: optional_string(&mut args, "--distributed-backend-id")?,
                reduce_scatter_parity_pass: optional_bool(
                    &mut args,
                    "--reduce-scatter-parity-pass",
                )?,
                all_gather_parity_pass: optional_bool(&mut args, "--all-gather-parity-pass")?,
                optimizer_update_parity_pass: optional_bool(
                    &mut args,
                    "--optimizer-update-parity-pass",
                )?,
                nccl_trace_validated: optional_bool(&mut args, "--nccl-trace-validated")?,
                overlap_validated: optional_bool(&mut args, "--overlap-validated")?,
                measured_comm_time_ms: optional_f64(&mut args, "--measured-comm-time-ms")?,
                measured_step_time_ms: optional_f64(&mut args, "--measured-step-time-ms")?,
                communication_speedup_x: optional_f64(&mut args, "--communication-speedup-x")?,
                generated_kernel_ids: optional_string(&mut args, "--generated-kernel-ids")?,
                generated_kernels: optional_bool(&mut args, "--generated-kernels")?,
                memory_reduction_x: optional_f64(&mut args, "--memory-reduction-x")?,
                trace_corpus_id: optional_string(&mut args, "--trace-corpus-id")?,
                calibration_report_id: optional_string(&mut args, "--calibration-report-id")?,
                fresh_profiler_traces: optional_bool(&mut args, "--fresh-profiler-traces")?,
                external_timing_validated: optional_bool(&mut args, "--external-timing-validated")?,
                holdout_spearman: optional_f64(&mut args, "--holdout-spearman")?,
                mean_abs_pct_error: optional_f64(&mut args, "--mean-abs-pct-error")?,
                evidence_source: optional_path(&mut args, "--evidence-source")?,
                source_report_dir: optional_path(&mut args, "--source-report-dir")?,
                evidence_id: optional_string(&mut args, "--evidence-id")?,
                generated_at: optional_string(&mut args, "--generated-at")?,
            };
            run_modelgolf_plan(options)?;
        }
        "backend-check" => {
            let backend = optional_string(&mut args, "--backend")?
                .as_deref()
                .map(parse_lite_backend)
                .transpose()?
                .unwrap_or(LiteBackendKind::MetalApple);
            let options = BackendCheckOptions {
                backend,
                output: optional_path(&mut args, "--output")?,
            };
            run_backend_check(options)?;
        }
        "wind-tunnel" => {
            if args.first().map(String::as_str).is_some_and(is_help_arg) {
                print_command_help("wind-tunnel");
                return Ok(());
            }
            match args.first().map(String::as_str) {
                Some("corpus") => {
                    args.remove(0);
                    let options = WindTunnelCorpusOptions {
                        spec: required_path(&mut args, "--spec")?,
                        trace_dir: required_path(&mut args, "--trace-dir")?,
                        output: optional_path(&mut args, "--output")?,
                    };
                    run_wind_tunnel_corpus(options)?;
                }
                Some("calibrate") => {
                    args.remove(0);
                    let options = WindTunnelCalibrationOptions {
                        spec: required_path(&mut args, "--spec")?,
                        trace_dir: optional_path(&mut args, "--trace-dir")?,
                        corpus: optional_path(&mut args, "--corpus")?,
                        output: optional_path(&mut args, "--output")?,
                    };
                    run_wind_tunnel_calibrate(options)?;
                }
                Some("scenario") => {
                    args.remove(0);
                    let options = WindTunnelScenarioOptions {
                        spec: required_path(&mut args, "--spec")?,
                        calibration: optional_path(&mut args, "--calibration")?,
                        trace: optional_path(&mut args, "--trace")?,
                        active_recurrent_replay_cut_ms: optional_f64(
                            &mut args,
                            "--active-recurrent-replay-cut-ms",
                        )?
                        .unwrap_or(10.0),
                        bank_update_cut_ms: optional_f64(&mut args, "--bank-update-cut-ms")?
                            .unwrap_or(0.0),
                        graph_overhead_cut_ms: optional_f64(&mut args, "--graph-overhead-cut-ms")?
                            .unwrap_or(0.0),
                        target_step_ms: optional_f64(&mut args, "--target-step-ms")?
                            .unwrap_or(120.0),
                        output: optional_path(&mut args, "--output")?,
                    };
                    run_wind_tunnel_scenario(options)?;
                }
                Some("report") => {
                    args.remove(0);
                    let options = WindTunnelReviewerOptions {
                        spec: required_path(&mut args, "--spec")?,
                        trace_dir: required_path(&mut args, "--trace-dir")?,
                        output_dir: required_path(&mut args, "--output-dir")?,
                        target_step_ms: optional_f64(&mut args, "--target-step-ms")?
                            .unwrap_or(120.0),
                        active_recurrent_replay_cut_ms: optional_f64(
                            &mut args,
                            "--active-recurrent-replay-cut-ms",
                        )?
                        .unwrap_or(10.0),
                    };
                    run_wind_tunnel_report(options)?;
                }
                _ => {
                    let options = WindTunnelOptions {
                        spec: required_path(&mut args, "--spec")?,
                        trace: optional_path(&mut args, "--trace")?,
                        output: optional_path(&mut args, "--output")?,
                    };
                    run_wind_tunnel(options)?;
                }
            }
        }
        "wind-tunnel-suite" => {
            let options = WindTunnelSuiteOptions {
                spec: required_path(&mut args, "--spec")?,
                trace_dir: required_path(&mut args, "--trace-dir")?,
                output_dir: required_path(&mut args, "--output-dir")?,
            };
            run_wind_tunnel_suite(options)?;
        }
        "trace-index" => {
            let options = TraceIndexOptions {
                spec: required_path(&mut args, "--spec")?,
                trace_dir: required_path(&mut args, "--trace-dir")?,
                output: optional_path(&mut args, "--output")?,
            };
            run_trace_index(options)?;
        }
        "lite" => {
            if args.first().map(String::as_str).is_some_and(is_help_arg) {
                print_command_help("lite");
                return Ok(());
            }
            let subcommand = args.first().map(String::as_str).ok_or_else(|| {
                PgError::InvalidOp(
                    "pg-local lite requires a subcommand: init, run, suite, benchmark, or verify-artifact"
                        .into(),
                )
            })?;
            if subcommand != "init"
                && subcommand != "run"
                && subcommand != "suite"
                && subcommand != "benchmark"
                && subcommand != "verify-artifact"
            {
                return Err(PgError::InvalidOp(format!(
                    "unknown pg-local lite subcommand: {subcommand}"
                )));
            }
            let subcommand = args.remove(0);
            if args.first().map(String::as_str).is_some_and(is_help_arg) {
                print_command_help(&format!("lite {subcommand}"));
                return Ok(());
            }
            match subcommand.as_str() {
                "init" => {
                    let backend = optional_string(&mut args, "--backend")?
                        .as_deref()
                        .map(parse_lite_backend)
                        .transpose()?
                        .unwrap_or(LiteBackendKind::CpuReference);
                    let options = LiteInitOptions {
                        input: required_path(&mut args, "--input")?,
                        output_dir: required_path(&mut args, "--output")?,
                        artifact_budget_bytes: optional_usize(
                            &mut args,
                            "--artifact-budget-bytes",
                        )?
                        .unwrap_or(1_000_000),
                        train_time_seconds: optional_f64(&mut args, "--train-time-seconds")?
                            .unwrap_or(300.0),
                        eval_time_seconds: optional_f64(&mut args, "--eval-time-seconds")?
                            .unwrap_or(60.0),
                        memory_budget_bytes: optional_usize(&mut args, "--memory-budget-bytes")?,
                        val_bytes: optional_usize(&mut args, "--val-bytes")?,
                        val_fraction: optional_f64(&mut args, "--val-fraction")?.unwrap_or(0.10),
                        context: optional_usize(&mut args, "--context")?.unwrap_or(512),
                        residual_buckets: optional_usize(&mut args, "--residual-buckets")?
                            .unwrap_or(2048),
                        residual_weight: optional_f64(&mut args, "--residual-weight")?
                            .unwrap_or(0.15),
                        backend,
                    };
                    run_lite_init(options)?;
                }
                "run" => {
                    let options = LiteRunOptions {
                        config: required_path(&mut args, "--config")?,
                        output: required_path(&mut args, "--output")?,
                    };
                    run_lite(options)?;
                }
                "suite" => {
                    let options = LiteSuiteOptions {
                        config_dir: required_path(&mut args, "--config-dir")?,
                        output_dir: required_path(&mut args, "--output")?,
                    };
                    run_lite_suite(options)?;
                }
                "benchmark" => {
                    let options = LiteBenchmarkOptions {
                        config: required_path(&mut args, "--config")?,
                        output_dir: required_path(&mut args, "--output")?,
                        repeats: optional_usize(&mut args, "--repeats")?.unwrap_or(3),
                    };
                    run_lite_benchmark(options)?;
                }
                "verify-artifact" => {
                    let options = LiteVerifyArtifactOptions {
                        artifact: required_path(&mut args, "--artifact")?,
                        config: optional_path(&mut args, "--config")?,
                        val: optional_path(&mut args, "--val")?,
                        output: optional_path(&mut args, "--output")?,
                    };
                    run_lite_verify_artifact(options)?;
                }
                _ => unreachable!(),
            }
        }
        other => {
            return Err(PgError::InvalidOp(format!(
                "unknown pg-local command: {other}"
            )));
        }
    }
    reject_extra_args(&args)?;
    Ok(())
}

fn print_help() {
    println!(
        "pg-local commands (JSON reports unless noted):\n\
         \n\
         pg-local verify --spec <path> [--artifact <path>] [--artifact-audit <json>] [--score-first-log <json>] [--output <json>]\n\
         pg-local proof-bundle --spec <path> --lite-config <path> [--lite-config-dir <dir>] --output-dir <dir> [--artifact <path>] [--trace <json>] [--trace-dir <dir>]\n\
         pg-local dist-sim --spec <path> [--world-size 8] [--steps 3] [--seed 1] [--output <json>]\n\
         pg-local artifact-lab --spec <path> [--artifact <path>] [--sweep quant,lqer,compression] [--mini-train <bytes>] [--mini-val <bytes>] [--output <json>]\n\
         pg-local modelgolf plan --spec <path> [--hardware local] [--context N] [--memory-budget-bytes N] [--artifact-budget-bytes N] [--proof-artifact <path>] [--quality-calibration <json>] [--release-evidence <json>] [--output <json>]\n\
         pg-local modelgolf pack|pack-experiment|pack-source-report|cache-plan|cache-experiment|cache-source-report|delta-plan|train-plan|kernel-plan|kernel-experiment|wind-plan|wind-experiment|scale-plan|release-check --spec <path> [planner options] [--proof-artifact <path>] [--quality-calibration <json>] [--release-evidence <json>] [--output <json>]\n\
         pg-local backend-check [--backend metal_apple] [--output <json>]\n\
         pg-local wind-tunnel --spec <path> [--trace <json>] [--output <json>]\n\
         pg-local wind-tunnel corpus --spec <path> --trace-dir <dir> [--output <json>]\n\
         pg-local wind-tunnel calibrate --spec <path> [--trace-dir <dir>|--corpus <json>] [--output <json>]\n\
         pg-local wind-tunnel scenario --spec <path> [--calibration <json>|--trace <json>] [--active-recurrent-replay-cut-ms 10] [--bank-update-cut-ms 0] [--graph-overhead-cut-ms 0] [--target-step-ms 120] [--output <json>]\n\
         pg-local wind-tunnel report --spec <path> --trace-dir <dir> --output-dir <dir> [--active-recurrent-replay-cut-ms 10] [--target-step-ms 120]\n\
         pg-local wind-tunnel-suite --spec <path> --trace-dir <dir> --output-dir <dir>\n\
         pg-local trace-index --spec <path> --trace-dir <dir> [--output <json>]\n\
         pg-local lite init --input <bytes> --output <dir> [--val-fraction 0.10] [--backend cpu_reference]\n\
         pg-local lite run --config <toml> --output <dir>\n\
         pg-local lite suite --config-dir <dir> --output <dir>\n\
         pg-local lite benchmark --config <toml> --output <dir> [--repeats 3]\n\
         pg-local lite verify-artifact --artifact <model.pglite.bin|json> [--config <toml>] [--val <bytes>] [--output <json>]\n\
         \n\
         Reports are written to --output when provided, otherwise to stdout.\n\
         wind-tunnel is estimate-only and PG-Lite is a local proxy benchmark;\n\
         neither is leaderboard or H100 validation evidence.\n\
         \n\
         Run 'pg-local help <command>' for command-specific notes."
    );
}

fn print_command_help(command: &str) {
    match command {
        "verify" => println!(
            "pg-local verify --spec <path> [--artifact <path>] [--artifact-audit <json>] [--score-first-log <json>] [--output <json>]\n\
             \n\
             Checks spec loading, artifact strict manifest compatibility when an artifact is supplied,\n\
             artifact byte budget, CaseOps sidecar structure, BPB smoke math, score-first legality,\n\
             and proposal claim boundaries. Missing optional artifacts are reported as skipped local\n\
             evidence, not as record proof."
        ),
        "proof-bundle" => println!(
            "pg-local proof-bundle --spec <path> --lite-config <path> [--lite-config-dir <dir>] --output-dir <dir> [--artifact <path>] [--trace <json>] [--trace-dir <dir>]\n\
             \n\
             Runs verify, dist-sim, artifact-lab, wind-tunnel, PG-Lite, PG-Lite suite, and backend checks into a local evidence\n\
             directory. The bundle is a non-record local proof packet; it does not establish final BPB,\n\
             H100 throughput, artifact byte proof from a record run, or leaderboard validity."
        ),
        "dist-sim" => println!(
            "pg-local dist-sim --spec <path> [--world-size 8] [--steps 3] [--seed 1] [--output <json>]\n\
             \n\
             Simulates reduce-scatter, shard-local Muon updates, all-gather reconstruction, optimizer\n\
             parity tolerance, and approximate communication bytes. It is deterministic local math\n\
             validation, not a network or NCCL benchmark."
        ),
        "artifact-lab" => println!(
            "pg-local artifact-lab --spec <path> [--artifact <path>] [--sweep quant,lqer,compression] [--mini-train <bytes>] [--mini-val <bytes>] [--output <json>]\n\
             \n\
             Emits quant layout CRCs, kernel IDs, group byte estimates, optional strict artifact load\n\
             status, artifact_kind, decoded PG-Lite artifact metadata when supplied, mixed bit-allocation\n\
             byte estimates, optional mini-BPB proxy, and synthetic proxy sweeps.\n\
             Sweep reconstruction and mini-BPB numbers are local proxies, not final compressed artifact or validation BPB evidence."
        ),
        "modelgolf" => println!(
            "pg-local modelgolf plan --spec <path> [--hardware <label>] [--runtime <label>] [--context N] [--batch N] [--memory-budget-bytes N] [--latency-target-ms X] [--quality-budget-ppl-pct X] [--artifact-budget-bytes N] [--delta-budget-bytes N] [--proof-artifact <path>] [--quality-calibration <json>] [--release-evidence <json>] [--output <json>]\n\
             pg-local modelgolf pack --spec <path> [--artifact-budget-bytes N] [--proof-artifact <path>] [--quality-calibration <json>] [--output <json>]\n\
             pg-local modelgolf pack-experiment --spec <path> [--artifact-budget-bytes N] [--output <json>]\n\
             pg-local modelgolf pack-source-report --spec <path> --release-artifact <path> --evidence-source <raw-evidence-json> --validation-dataset-id <id> --validation-command <cmd> --heldout-bpb X --baseline-bpb X --decode-tokens-per-second X [--hardware <label>] [--runtime <label>] [--quality-budget-ppl-pct X] [--output <json>]\n\
             pg-local modelgolf lqer-source-report --spec <path> --evidence-source <raw-evidence-json> --calibration-dataset-id <id> --production-svd-validated true --calibrated-tensor-sensitivity true --equal-byte-bpb-delta X [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf cache-plan --spec <path> [--context N] [--batch N] [--memory-budget-bytes N] [--output <json>]\n\
             pg-local modelgolf cache-experiment --spec <path> [--context N] [--batch N] [--output <json>]\n\
             pg-local modelgolf cache-source-report --spec <path> --context N --batch N --memory-budget-bytes N --evidence-source <raw-evidence-json> --backend-id <id> --kernel-id <id> --long-context-dataset-id <id> --fused-runtime true --parity-pass true --long-context-bpb-delta-pct X --speedup-x X [--quality-budget-ppl-pct X] [--output <json>]\n\
             pg-local modelgolf delta-plan --spec <path> [--delta-budget-bytes N] [--output <json>]\n\
             pg-local modelgolf delta-source-report --spec <path> --delta-budget-bytes N --evidence-source <raw-evidence-json> --domain-dataset-id <id> --trained-delta-bytes N --legality-pass true --score-first-trace-or-review true --equal-byte-domain-bpb-delta X [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf train-plan --spec <path> [--output <json>]\n\
             pg-local modelgolf train-source-report --spec <path> --evidence-source <raw-evidence-json> --training-run-id <id> --backend-id <id> --gpu-backend-integrated true --proxy-calibrated true --post-export-bpb-delta-vs-posthoc X [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf resource-cost-source-report --spec <path> --evidence-source <raw-evidence-json> --measurement-run-id <id> --power-meter-id <id> --wall-time-seconds X --average-power-watts X --energy-joules X --telemetry-validated true [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf optimizer-comm-source-report --spec <path> --evidence-source <raw-evidence-json> --distributed-backend-id <id> --reduce-scatter-parity-pass true --all-gather-parity-pass true --optimizer-update-parity-pass true --nccl-trace-validated true --overlap-validated true --measured-comm-time-ms X --measured-step-time-ms X --communication-speedup-x X [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf kernel-plan --spec <path> [--output <json>]\n\
             pg-local modelgolf kernel-experiment --spec <path> [--output <json>]\n\
             pg-local modelgolf kernel-source-report --spec <path> --evidence-source <raw-evidence-json> --backend-id <id> --generated-kernel-ids <id[,id]> --generated-kernels true --parity-pass true --speedup-x X --memory-reduction-x X [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf wind-plan --spec <path> [--output <json>]\n\
             pg-local modelgolf wind-experiment --spec <path> [--context N] [--batch N] [--memory-budget-bytes N] [--artifact-budget-bytes N] [--output <json>]\n\
             pg-local modelgolf wind-source-report --spec <path> --evidence-source <raw-evidence-json> --trace-corpus-id <id> --calibration-report-id <id> --fresh-profiler-traces true --external-timing-validated true --holdout-spearman X --mean-abs-pct-error X [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf scale-plan --spec <path> [--hardware <label>] [--runtime <label>] [--context N] [--memory-budget-bytes N] [--artifact-budget-bytes N] [--output <json>]\n\
             pg-local modelgolf release-evidence --spec <path> --source-report-dir <dir> --evidence-id <id> [--generated-at <timestamp>] [--hardware <label>] [--runtime <label>] [--output <json>]\n\
             pg-local modelgolf release-check --spec <path> [--release-evidence <json>] [--output <json>]\n\
             \n\
             Emits the constraint-native ModelGolf planner surface: ResourceContract, ModelArtifactIR,\n\
             mixed-precision/LQER Pack plan, CacheGolf KV plan, DeltaGolf byte-constrained adapter plan,\n\
             TrainGolf artifact-aware objective plus CPU train-loop seam, Optimizer/Comm compiler report,\n\
             KernelForge fusion contracts, Wind Tunnel Pareto summary, and ScaleGolf track matrix. When --proof-artifact is supplied, Pack also exports,\n\
             strict-reloads, and smoke-scores a deterministic local artifact for byte/reload evidence.\n\
             pack-experiment runs the plan's LQER++ table on a bounded deterministic small model:\n\
             uniform Q4, mixed Q4/Q6, mixed+LQER top-3, and deterministic random-LQER control.\n\
             pack-source-report emits a SHA-256 bound modelgolf_release_source_report for Pack after strict-reloading\n\
             --release-artifact and binding caller-supplied held-out BPB/decode metrics plus a semantically checked --evidence-source modelgolf_raw_evidence packet to the current spec/runtime.\n\
             lqer-source-report emits a modelgolf_release_source_report for LQER++ by binding measured\n\
             production SVD/calibration/equal-byte quality claims to the current selected LQER plan.\n\
             cache-experiment runs the plan's K/V bit-grid protocol on deterministic attention:\n\
             measured output error, perturbation-bound coverage, bound tightness, and long-context memory rows.\n\
             cache-source-report emits a modelgolf_release_source_report for CacheGolf by binding measured\n\
             fused-runtime long-context quality/timing claims to the selected KV policy for the current contract.\n\
             delta/train/resource-cost/optimizer-comm/kernel/wind source-report commands emit SHA-bound release source reports for their\n\
             corresponding measured domain-delta, backend-training, energy telemetry, distributed optimizer, generated-kernel, and trace-calibration claims.\n\
             release-evidence assembles a modelgolf_release_evidence JSON file from all nine source reports\n\
             in --source-report-dir, recomputing SHA-256 and revalidating each source report's raw evidence packet.\n\
             kernel-experiment runs the plan's exact tiled CE protocol on a deterministic CPU fixture:\n\
             full logits vs tiled logit-free CE, loss/dH/dW parity, CE scratch bytes, and CPU timing.\n\
             scale-plan emits the ScaleGolf track matrix for PG-Lite, ParameterGolf, DeltaGolf, LongContext, and ScaleGolf using the same resource contract/model IR/artifact/runtime/evaluator/cost surfaces.\n\
             release-check emits a fail-closed checklist of remaining release blockers and required external evidence.\n\
             When --release-evidence is supplied, release-check validates measured artifact, quality, runtime, and trace evidence against the checklist.\n\
             When --quality-calibration is supplied, Pack scales role/group bit-loss estimates from measured JSON points.\n\
             This is deterministic local planning evidence, not final H100 timing or BPB validation."
        ),
        "backend-check" => println!(
            "pg-local backend-check [--backend cpu_reference|metal_apple|mlx_prototype] [--output <json>]\n\
             \n\
             Reports local backend readiness without running a benchmark. For metal_apple it checks\n\
             the checked-in PG-Lite Metal kernel source, `xcrun --find metal`, Cargo feature state,\n\
             and whether the Rust runtime is linked. This is a readiness probe, not acceleration evidence."
        ),
        "wind-tunnel" => println!(
            "pg-local wind-tunnel --spec <path> [--trace <json>] [--output <json>]\n\
             pg-local wind-tunnel corpus --spec <path> --trace-dir <dir> [--output <json>]\n\
             pg-local wind-tunnel calibrate --spec <path> [--trace-dir <dir>|--corpus <json>] [--output <json>]\n\
             pg-local wind-tunnel scenario --spec <path> [--calibration <json>|--trace <json>] [--active-recurrent-replay-cut-ms 10] [--bank-update-cut-ms 0] [--graph-overhead-cut-ms 0] [--target-step-ms 120] [--output <json>]\n\
             pg-local wind-tunnel report --spec <path> --trace-dir <dir> --output-dir <dir> [--active-recurrent-replay-cut-ms 10] [--target-step-ms 120]\n\
             \n\
             Emits estimate_only=true, stage estimates, optional trace calibration, expected steps in\n\
             600s, rough eval/artifact estimates, bottleneck hints, ranked experiment candidates,\n\
             deduped historical trace corpus reports, calibration models, and scenario reports,\n\
             plus a reviewer-facing report command that bundles trace index, holdout error,\n\
             and a spec-derived operation DAG.\n\
             This is a planning model only; it is not an H100 emulator or proof of record timing."
        ),
        "wind-tunnel-suite" => println!(
            "pg-local wind-tunnel-suite --spec <path> --trace-dir <dir> --output-dir <dir>\n\
             \n\
             Runs Wind Tunnel over JSON/log traces in <dir>, writes one report per trace plus\n\
             summary.json and summary.md, and compares trace totals/stage attribution against the shape-model baseline.\n\
             The suite is for calibration and triage; it is not H100 validation evidence."
        ),
        "trace-index" => println!(
            "pg-local trace-index --spec <path> --trace-dir <dir> [--output <json>]\n\
             \n\
             Recursively scans historical JSON/log/markdown trace artifacts, extracts timing and audit metadata,\n\
             classifies total-only versus stage-attribution traces, tags exact/#2135/canonical/active-recurrent evidence,\n\
             and reports fastest usable timing records. This is an evidence index, not validation by itself."
        ),
        "lite" => println!(
            "pg-local lite init --input <bytes> --output <dir> [--artifact-budget-bytes 1000000] [--val-fraction 0.10] [--val-bytes N] [--backend cpu_reference]\n\
             pg-local lite run --config <toml> --output <dir>\n\
             pg-local lite suite --config-dir <dir> --output <dir>\n\
             pg-local lite benchmark --config <toml> --output <dir> [--repeats 3]\n\
             pg-local lite verify-artifact --artifact <model.pglite.bin|json> [--config <toml>] [--val <bytes>] [--output <json>]\n\
             \n\
             Initializes, runs, verifies, or benchmarks PG-Lite configs. PG-Lite BPB is local\n\
             proxy evidence only."
        ),
        "lite init" => println!(
            "pg-local lite init --input <bytes> --output <dir> [--artifact-budget-bytes 1000000] [--train-time-seconds 300] [--eval-time-seconds 60] [--memory-budget-bytes N] [--val-fraction 0.10|--val-bytes N] [--context 512] [--residual-buckets 2048] [--residual-weight 0.15] [--backend cpu_reference|metal_apple]\n\
             \n\
             Creates a self-contained local PG-Lite benchmark directory with train/validation byte splits,\n\
             byte_golf residual and n-gram baseline configs, stream_golf score-first config,\n\
             artifact_golf baseline config, init_report.json, and README.md."
        ),
        "lite run" => println!(
            "pg-local lite run --config <toml> --output <dir>\n\
             \n\
             Runs a PG-Lite benchmark and writes <dir>/report.json. Supported configs\n\
             use score=\"bpb\", data.format=\"bytes\", and vocab=\"byte\". backend.kind=\"cpu_reference\"\n\
             is portable. backend.kind=\"metal_apple\" executes the feature-gated Metal backend when\n\
             built with --features metal_apple and a device is visible; otherwise it fails unless\n\
             [backend].allow_cpu_fallback=true. backend.kind=\"mlx_prototype\" is a parsed future\n\
             interface and requires explicit CPU fallback.\n\
             Tracks: byte_golf, stream_golf, artifact_golf. Outputs include report.json,\n\
             model.pglite.json, and artifact_manifest.json with train/eval wall time, estimated\n\
             and actual artifact bytes, time/artifact/memory budget booleans, validation_bpb,\n\
             score-first counters, proxy_only=true, and status. artifact_golf may load\n\
             [model].artifact_path. PG-Lite BPB is local proxy evidence only."
        ),
        "lite suite" => println!(
            "pg-local lite suite --config-dir <dir> --output <dir>\n\
             \n\
             Runs every *.toml config in <dir>, writing one output directory per config plus\n\
             summary.json and summary.md. The summary ranks local proxy BPB while preserving per-track\n\
             budget and score-first legality fields."
        ),
        "lite benchmark" => println!(
            "pg-local lite benchmark --config <toml> --output <dir> [--repeats 3]\n\
             \n\
             Runs the same PG-Lite config with cpu_reference and metal_apple requested backends,\n\
             with CPU fallback disabled so acceleration failures are visible. Writes per-run reports\n\
             plus summary.json and summary.md with median wall time, train/eval split, BPB parity,\n\
             artifact bytes, and CPU-vs-Metal speedup where Metal is available."
        ),
        "lite verify-artifact" => println!(
            "pg-local lite verify-artifact --artifact <model.pglite.bin|json> [--config <toml>] [--val <bytes>] [--output <json>]\n\
             \n\
             Strict-loads a PG-Lite artifact, verifies its decoded model fingerprint, checks the\n\
             supplied config fingerprint and artifact budget when --config is present, and optionally\n\
             computes local proxy BPB on --val or the config validation split."
        ),
        other => println!("No command-specific help for '{other}'. Run 'pg-local --help'."),
    }
}

fn is_help_arg(arg: &str) -> bool {
    arg == "--help" || arg == "-h"
}

fn required_path(args: &mut Vec<String>, flag: &str) -> PgResult<PathBuf> {
    optional_path(args, flag)?.ok_or_else(|| PgError::InvalidOp(format!("{flag} is required")))
}

fn optional_path(args: &mut Vec<String>, flag: &str) -> PgResult<Option<PathBuf>> {
    Ok(optional_string(args, flag)?.map(PathBuf::from))
}

fn optional_usize(args: &mut Vec<String>, flag: &str) -> PgResult<Option<usize>> {
    optional_string(args, flag)?
        .map(|raw| {
            raw.parse::<usize>()
                .map_err(|err| PgError::InvalidOp(format!("{flag} expects usize: {err}")))
        })
        .transpose()
}

fn optional_u64(args: &mut Vec<String>, flag: &str) -> PgResult<Option<u64>> {
    optional_string(args, flag)?
        .map(|raw| {
            raw.parse::<u64>()
                .map_err(|err| PgError::InvalidOp(format!("{flag} expects u64: {err}")))
        })
        .transpose()
}

fn optional_f64(args: &mut Vec<String>, flag: &str) -> PgResult<Option<f64>> {
    optional_string(args, flag)?
        .map(|raw| {
            raw.parse::<f64>()
                .map_err(|err| PgError::InvalidOp(format!("{flag} expects f64: {err}")))
        })
        .transpose()
}

fn optional_bool(args: &mut Vec<String>, flag: &str) -> PgResult<Option<bool>> {
    optional_string(args, flag)?
        .map(|raw| match raw.as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" => Ok(false),
            _ => Err(PgError::InvalidOp(format!(
                "{flag} expects bool: true/false"
            ))),
        })
        .transpose()
}

fn optional_string(args: &mut Vec<String>, flag: &str) -> PgResult<Option<String>> {
    if let Some(pos) = args.iter().position(|arg| arg == flag) {
        args.remove(pos);
        if pos >= args.len() {
            return Err(PgError::InvalidOp(format!("{flag} requires a value")));
        }
        Ok(Some(args.remove(pos)))
    } else {
        Ok(None)
    }
}

fn parse_lite_backend(raw: &str) -> PgResult<LiteBackendKind> {
    match raw {
        "cpu_reference" => Ok(LiteBackendKind::CpuReference),
        "metal_apple" => Ok(LiteBackendKind::MetalApple),
        "mlx_prototype" => Ok(LiteBackendKind::MlxPrototype),
        other => Err(PgError::InvalidOp(format!(
            "unknown PG-Lite backend {other}; expected cpu_reference, metal_apple, or mlx_prototype"
        ))),
    }
}

fn parse_modelgolf_section(args: &mut Vec<String>) -> PgResult<ModelGolfSection> {
    let Some(first) = args.first().map(String::as_str) else {
        return Ok(ModelGolfSection::Full);
    };
    if first.starts_with("--") {
        return Ok(ModelGolfSection::Full);
    }
    let section = match first {
        "plan" => ModelGolfSection::Full,
        "pack" => ModelGolfSection::Pack,
        "pack-experiment" => ModelGolfSection::PackExperiment,
        "pack-source-report" => ModelGolfSection::PackSourceReport,
        "lqer-source-report" => ModelGolfSection::LqerSourceReport,
        "cache-plan" => ModelGolfSection::Cache,
        "cache-experiment" => ModelGolfSection::CacheExperiment,
        "cache-source-report" => ModelGolfSection::CacheSourceReport,
        "delta-plan" => ModelGolfSection::Delta,
        "delta-source-report" => ModelGolfSection::DeltaSourceReport,
        "train-plan" => ModelGolfSection::Train,
        "train-source-report" => ModelGolfSection::TrainSourceReport,
        "resource-cost-source-report" => ModelGolfSection::ResourceCostSourceReport,
        "optimizer-comm-source-report" => ModelGolfSection::OptimizerCommSourceReport,
        "kernel-plan" => ModelGolfSection::Kernel,
        "kernel-experiment" => ModelGolfSection::KernelExperiment,
        "kernel-source-report" => ModelGolfSection::KernelSourceReport,
        "wind-plan" => ModelGolfSection::Wind,
        "wind-experiment" => ModelGolfSection::WindExperiment,
        "wind-source-report" => ModelGolfSection::WindSourceReport,
        "scale-plan" => ModelGolfSection::Scale,
        "release-evidence" => ModelGolfSection::ReleaseEvidence,
        "release-check" => ModelGolfSection::ReleaseCheck,
        other => {
            return Err(PgError::InvalidOp(format!(
                "unknown modelgolf subcommand: {other}; expected plan, pack, pack-experiment, pack-source-report, lqer-source-report, cache-plan, cache-experiment, cache-source-report, delta-plan, delta-source-report, train-plan, train-source-report, resource-cost-source-report, optimizer-comm-source-report, kernel-plan, kernel-experiment, kernel-source-report, wind-plan, wind-experiment, wind-source-report, scale-plan, release-evidence, or release-check"
            )));
        }
    };
    args.remove(0);
    Ok(section)
}

fn reject_extra_args(args: &[String]) -> PgResult<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(PgError::InvalidOp(format!(
            "unexpected pg-local arguments: {}",
            args.join(" ")
        )))
    }
}
