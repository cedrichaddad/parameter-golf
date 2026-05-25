# Claim Spine

1. Parameter Golf is interesting because algorithm quality, artifact bytes, runtime, data movement, and legal evaluation all constrain the same design.
2. The systems work is not a generic "make PyTorch faster" story; it is a case study in removing overhead that becomes model architecture pressure.
3. The stack now has credible instrumentation: device-resident batches, zero measured BF16/F32 backward bridges, graph-captured no-loss backward, and post-export audit surfaces.
4. Timing improved from an unusable real path to a clean exact proxy band near the target, with one diagnostic all-ST profile below the time target.
5. The honest current state is that speed work exposed the next blocker: BPB/export/recurrence correctness, not lack of timing instrumentation.
6. The final roadmap is narrow: exact active recurrent replay, full record train/eval/export, artifact proof, and full-validation BPB.

