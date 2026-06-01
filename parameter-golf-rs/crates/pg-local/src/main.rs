use std::path::PathBuf;

use pg_core::{PgError, PgResult};
use pg_local::{
    ArtifactLabOptions, BackendCheckOptions, DistSimOptions, LiteBackendKind, LiteRunOptions,
    LiteSuiteOptions, ProofBundleOptions, TraceIndexOptions, VerifyOptions, WindTunnelOptions,
    WindTunnelSuiteOptions, run_artifact_lab, run_backend_check, run_dist_sim, run_lite,
    run_lite_suite, run_proof_bundle, run_trace_index, run_verify, run_wind_tunnel,
    run_wind_tunnel_suite,
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
        if let Some(topic) = args.first() {
            print_command_help(topic);
        } else {
            print_help();
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
            let options = WindTunnelOptions {
                spec: required_path(&mut args, "--spec")?,
                trace: optional_path(&mut args, "--trace")?,
                output: optional_path(&mut args, "--output")?,
            };
            run_wind_tunnel(options)?;
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
                PgError::InvalidOp("pg-local lite requires a subcommand: run or suite".into())
            })?;
            if subcommand != "run" && subcommand != "suite" {
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
         pg-local proof-bundle --spec <path> --lite-config <path> [--lite-config-dir <dir>] --output-dir <dir> [--artifact <path>] [--trace <json>]\n\
         pg-local dist-sim --spec <path> [--world-size 8] [--steps 3] [--seed 1] [--output <json>]\n\
         pg-local artifact-lab --spec <path> [--artifact <path>] [--sweep quant,lqer,compression] [--mini-train <bytes>] [--mini-val <bytes>] [--output <json>]\n\
         pg-local backend-check [--backend metal_apple] [--output <json>]\n\
         pg-local wind-tunnel --spec <path> [--trace <json>] [--output <json>]\n\
         pg-local wind-tunnel-suite --spec <path> --trace-dir <dir> --output-dir <dir>\n\
         pg-local trace-index --spec <path> --trace-dir <dir> [--output <json>]\n\
         pg-local lite run --config <toml> --output <dir>\n\
         pg-local lite suite --config-dir <dir> --output <dir>\n\
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
            "pg-local proof-bundle --spec <path> --lite-config <path> [--lite-config-dir <dir>] --output-dir <dir> [--artifact <path>] [--trace <json>]\n\
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
        "backend-check" => println!(
            "pg-local backend-check [--backend cpu_reference|metal_apple|mlx_prototype] [--output <json>]\n\
             \n\
             Reports local backend readiness without running a benchmark. For metal_apple it checks\n\
             the checked-in PG-Lite Metal kernel source, `xcrun --find metal`, Cargo feature state,\n\
             and whether the Rust runtime is linked. This is a readiness probe, not acceleration evidence."
        ),
        "wind-tunnel" => println!(
            "pg-local wind-tunnel --spec <path> [--trace <json>] [--output <json>]\n\
             \n\
             Emits estimate_only=true, stage estimates, optional trace calibration, expected steps in\n\
             600s, rough eval/artifact estimates, bottleneck hints, ranked experiment candidates,\n\
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
            "pg-local lite run --config <toml> --output <dir>\n\
             pg-local lite suite --config-dir <dir> --output <dir>\n\
             \n\
             Runs a single PG-Lite config or the full local config directory. PG-Lite BPB is local\n\
             proxy evidence only."
        ),
        "lite run" => println!(
            "pg-local lite run --config <toml> --output <dir>\n\
             \n\
             Runs the PG-Lite CPU reference benchmark and writes <dir>/report.json. Supported configs\n\
             use score=\"bpb\", data.format=\"bytes\", vocab=\"byte\", and backend.kind=\"cpu_reference\".\n\
             metal_apple and mlx_prototype are interface probes only: they fail unless\n\
             [backend].allow_cpu_fallback=true, and fallback reports backend_accelerated=false.\n\
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
