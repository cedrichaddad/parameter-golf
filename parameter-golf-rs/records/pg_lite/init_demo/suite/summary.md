# PG-Lite Suite

- Configs: `5`
- Status: `pass`
- Best local proxy BPB: `7.161766` from `records/pg_lite/init_demo/configs/byte_golf_ngram_baseline.toml`

| Config | Track | Family | Backend | Status | BPB | Artifact bytes | Artifact budget | Score-first updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `records/pg_lite/init_demo/configs/artifact_golf_empty.toml` | `artifact_golf` | `artifact_only` | `cpu_reference` -> `cpu_reference` | `pass` | 8.000000 | 96 | `pass` | 0/0 |
| `records/pg_lite/init_demo/configs/byte_golf_metal_probe.toml` | `byte_golf` | `ngram_residual` | `metal_apple` -> `cpu_reference` | `pass` | 7.240209 | 2098 | `pass` | 0/0 |
| `records/pg_lite/init_demo/configs/byte_golf_ngram_baseline.toml` | `byte_golf` | `byte_ngram` | `cpu_reference` -> `cpu_reference` | `pass` | 7.161766 | 1062 | `pass` | 0/0 |
| `records/pg_lite/init_demo/configs/byte_golf_ngram_residual.toml` | `byte_golf` | `ngram_residual` | `cpu_reference` -> `cpu_reference` | `pass` | 7.240209 | 2098 | `pass` | 0/0 |
| `records/pg_lite/init_demo/configs/stream_golf_score_first.toml` | `stream_golf` | `ngram_residual` | `cpu_reference` -> `cpu_reference` | `pass` | 7.205932 | 2098 | `pass` | 43/43 |

PG-Lite BPB is local proxy evidence only; it is not Parameter Golf leaderboard evidence.
