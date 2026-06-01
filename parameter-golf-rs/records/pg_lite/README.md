# PG-Lite Local Baseline Records

This folder contains checked-in PG-Lite CPU-reference baseline runs. PG-Lite is
a local proxy benchmark, not the official Parameter Golf leaderboard benchmark.
Do not compare these BPB values with FineWeb/H100 record BPB.

Regenerate the suite from `parameter-golf-rs/` with:

```bash
cargo run -q -p pg-local -- lite suite \
  --config-dir specs/pg_lite \
  --output records/pg_lite/suite
```

## Current Suite

- Configs: 5
- Status: pass
- Best local proxy BPB: 6.894108
- Best config: `specs/pg_lite/byte_golf_ngram_baseline.toml`

The suite writes one directory per config with:

- `report.json`
- `artifact_manifest.json`
- `model.pglite.bin`
- `model.pglite.json`

The `artifact_actual_bytes` field is the actual serialized local
`model.pglite.bin` byte count and is used for the PG-Lite budget gate. The JSON
artifact is retained as a stable debug representation. Neither artifact is the
official Rust/CUDA compressed artifact proof.

## Current Results

| Config | Track | Family | BPB | Artifact Bytes | Score-First Updates |
|---|---|---|---:|---:|---:|
| `artifact_golf_1mb.toml` | artifact_golf | artifact_only | 8.000000 | 96 | 0 |
| `byte_golf_1mb_300s.toml` | byte_golf | ngram_residual | 6.975617 | 2497 | 0 |
| `byte_golf_256kb_60s.toml` | byte_golf | ngram_residual | 6.957108 | 2490 | 0 |
| `byte_golf_ngram_baseline.toml` | byte_golf | byte_ngram | 6.894108 | 1181 | 0 |
| `stream_golf_1mb_score_first.toml` | stream_golf | ngram_residual | 6.940747 | 2497 | 82 |

These sample-corpus numbers are regression checks. They are useful for catching
local evaluator, artifact, and score-first-ordering regressions before another
paid H100 run.
