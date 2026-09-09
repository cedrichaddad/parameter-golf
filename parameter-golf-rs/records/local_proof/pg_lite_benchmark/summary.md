# PG-Lite Backend Benchmark

- Config: `specs/pg_lite/byte_golf_1mb_300s_metal_probe.toml`
- Repeats: `3`
- Status: `pass`
- CPU-vs-Metal total wall speedup: `0.963x`
- CPU-vs-Metal train wall speedup: `0.040x`
- Metal minus CPU local BPB delta: `0.000000062`
- Fastest local backend: `cpu_reference`
- Best local proxy BPB backend: `cpu_reference`

| Requested backend | Status | Runs | Accelerated | Execution backends | Median total s | Median train s | Median eval s | Median BPB | Artifact bytes | Error |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `cpu_reference` | `pass` | 3 | 0/3 | `cpu_reference` | 0.150236 | 0.000100 | 0.000336 | 6.975616794 | 2497 |  |
| `metal_apple` | `pass` | 3 | 3/3 | `metal_apple` | 0.156023 | 0.002517 | 0.000000 | 6.975616856 | 2497 |  |

PG-Lite benchmark results are local proxy evidence. They do not establish H100 throughput, final BPB, or leaderboard validity.
