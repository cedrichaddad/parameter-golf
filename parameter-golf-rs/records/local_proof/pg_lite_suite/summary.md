# PG-Lite Suite

- Configs: `5`
- Status: `pass`
- Best local proxy BPB: `6.894108` from `specs/pg_lite/byte_golf_ngram_baseline.toml`

| Config | Track | Family | Backend | Status | BPB | Artifact bytes | Artifact budget | Score-first updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `specs/pg_lite/artifact_golf_1mb.toml` | `artifact_golf` | `artifact_only` | `cpu_reference` -> `cpu_reference` | `pass` | 8.000000 | 96 | `pass` | 0/0 |
| `specs/pg_lite/byte_golf_1mb_300s.toml` | `byte_golf` | `ngram_residual` | `cpu_reference` -> `cpu_reference` | `pass` | 6.975617 | 2497 | `pass` | 0/0 |
| `specs/pg_lite/byte_golf_256kb_60s.toml` | `byte_golf` | `ngram_residual` | `cpu_reference` -> `cpu_reference` | `pass` | 6.957108 | 2490 | `pass` | 0/0 |
| `specs/pg_lite/byte_golf_ngram_baseline.toml` | `byte_golf` | `byte_ngram` | `cpu_reference` -> `cpu_reference` | `pass` | 6.894108 | 1181 | `pass` | 0/0 |
| `specs/pg_lite/stream_golf_1mb_score_first.toml` | `stream_golf` | `ngram_residual` | `cpu_reference` -> `cpu_reference` | `pass` | 6.940747 | 2497 | `pass` | 82/82 |

PG-Lite BPB is local proxy evidence only; it is not Parameter Golf leaderboard evidence.
