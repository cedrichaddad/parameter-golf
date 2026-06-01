/// BigramHash — learned hash-based lexical memory.
///
/// XOR-hash two consecutive token IDs into a bucket, look up a learned embedding.
/// BigramHash(1536 buckets, 128 embed_dim) → 197K params.
///
/// Matches SOTA Python: hash = (36313 * cur) XOR (27191 * prev) % (num_buckets - 1)
/// Position 0 maps to (num_buckets - 1) as a sentinel bucket.

const HASH_A: u32 = 36313;
const HASH_B: u32 = 27191;

/// Compute bigram hash bucket index (SOTA convention).
/// For t=0 (no previous token), returns num_buckets - 1 (sentinel).
#[inline]
pub fn bigram_hash(prev: Option<u32>, cur: u32, num_buckets: usize) -> usize {
    match prev {
        None => num_buckets - 1,
        Some(p) => {
            let h = cur.wrapping_mul(HASH_A) ^ p.wrapping_mul(HASH_B);
            (h as usize) % (num_buckets - 1)
        }
    }
}

/// Compute the fused CUDA kernel bucket for a flat [batch, seq] token buffer.
/// Each sequence boundary uses the sentinel bucket rather than the previous
/// sequence's last token.
#[inline]
pub fn bigram_hash_flat_position(
    tokens: &[u32],
    t: usize,
    seq_len: usize,
    num_buckets: usize,
) -> usize {
    let prev = if t % seq_len == 0 {
        None
    } else {
        Some(tokens[t - 1])
    };
    bigram_hash(prev, tokens[t], num_buckets)
}

/// Forward: compute bigram embeddings for a sequence.
/// tokens: [seq_len]
/// embedding_table: [num_buckets, embed_dim]
/// output: [seq_len, embed_dim]
pub fn bigram_hash_forward(
    tokens: &[u32],
    embedding_table: &[f32],
    output: &mut [f32],
    num_buckets: usize,
    embed_dim: usize,
) {
    let seq_len = tokens.len();
    for t in 0..seq_len {
        let prev = if t == 0 { None } else { Some(tokens[t - 1]) };
        let bucket = bigram_hash(prev, tokens[t], num_buckets);
        let src = &embedding_table[bucket * embed_dim..(bucket + 1) * embed_dim];
        let dst = &mut output[t * embed_dim..(t + 1) * embed_dim];
        dst.copy_from_slice(src);
    }
}

/// Backward: accumulate gradients into embedding table.
/// grad_output: [seq_len, embed_dim]
/// grad_embedding: [num_buckets, embed_dim] (accumulated, NOT zeroed here)
pub fn bigram_hash_backward(
    tokens: &[u32],
    grad_output: &[f32],
    grad_embedding: &mut [f32],
    num_buckets: usize,
    embed_dim: usize,
) {
    let seq_len = tokens.len();
    for t in 0..seq_len {
        let prev = if t == 0 { None } else { Some(tokens[t - 1]) };
        let bucket = bigram_hash(prev, tokens[t], num_buckets);
        let go = &grad_output[t * embed_dim..(t + 1) * embed_dim];
        let ge = &mut grad_embedding[bucket * embed_dim..(bucket + 1) * embed_dim];
        for d in 0..embed_dim {
            ge[d] += go[d];
        }
    }
}

/// CPU reference for the fused token embedding + BigramHash projection CUDA path.
pub fn embedding_bigram_project_merge_forward(
    ids: &[u32],
    tok_emb: &[f32],
    bigram_embed: &[f32],
    bigram_proj: &[f32],
    bigram_scale: f32,
    out: &mut [f32],
    model_dim: usize,
    bigram_dim: usize,
    bigram_vocab: usize,
    seq_len: usize,
) {
    assert_eq!(tok_emb.len() % model_dim, 0);
    assert_eq!(bigram_embed.len(), bigram_vocab * bigram_dim);
    assert_eq!(bigram_proj.len(), model_dim * bigram_dim);
    assert_eq!(out.len(), ids.len() * model_dim);
    for t in 0..ids.len() {
        let tok = ids[t] as usize;
        let bucket = bigram_hash_flat_position(ids, t, seq_len, bigram_vocab);
        for j in 0..model_dim {
            let mut acc = 0.0f32;
            for k in 0..bigram_dim {
                acc += bigram_embed[bucket * bigram_dim + k] * bigram_proj[j * bigram_dim + k];
            }
            out[t * model_dim + j] = tok_emb[tok * model_dim + j] + bigram_scale * acc;
        }
    }
}

/// CPU reference for the fused BigramHash merge backward CUDA path.
#[allow(clippy::too_many_arguments)]
pub fn embedding_bigram_project_merge_backward(
    ids: &[u32],
    grad_out: &[f32],
    bigram_embed: &[f32],
    bigram_proj: &[f32],
    bigram_scale: f32,
    grad_tok_emb: &mut [f32],
    grad_bigram_embed: &mut [f32],
    grad_bigram_proj: &mut [f32],
    grad_bigram_scale: &mut f32,
    model_dim: usize,
    bigram_dim: usize,
    bigram_vocab: usize,
    seq_len: usize,
) {
    assert_eq!(grad_out.len(), ids.len() * model_dim);
    assert_eq!(bigram_embed.len(), bigram_vocab * bigram_dim);
    assert_eq!(bigram_proj.len(), model_dim * bigram_dim);
    assert_eq!(grad_bigram_embed.len(), bigram_vocab * bigram_dim);
    assert_eq!(grad_bigram_proj.len(), model_dim * bigram_dim);
    for t in 0..ids.len() {
        let tok = ids[t] as usize;
        let bucket = bigram_hash_flat_position(ids, t, seq_len, bigram_vocab);
        for j in 0..model_dim {
            let go = grad_out[t * model_dim + j];
            grad_tok_emb[tok * model_dim + j] += go;
            let mut projected = 0.0f32;
            for k in 0..bigram_dim {
                let e = bigram_embed[bucket * bigram_dim + k];
                let p = bigram_proj[j * bigram_dim + k];
                projected += e * p;
                grad_bigram_proj[j * bigram_dim + k] += bigram_scale * go * e;
            }
            *grad_bigram_scale += go * projected;
        }
        for k in 0..bigram_dim {
            let mut acc = 0.0f32;
            for j in 0..model_dim {
                acc += grad_out[t * model_dim + j] * bigram_proj[j * bigram_dim + k];
            }
            grad_bigram_embed[bucket * bigram_dim + k] += bigram_scale * acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigram_hash_deterministic() {
        let h1 = bigram_hash(Some(100), 200, 1536);
        let h2 = bigram_hash(Some(100), 200, 1536);
        assert_eq!(h1, h2);
        assert!(h1 < 1536);
    }

    #[test]
    fn test_bigram_hash_different_inputs() {
        let h1 = bigram_hash(Some(100), 200, 1536);
        let h2 = bigram_hash(Some(200), 100, 1536);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_bigram_hash_sentinel() {
        // Position 0 (no prev) should map to sentinel bucket
        let h = bigram_hash(None, 42, 1536);
        assert_eq!(h, 1535);
    }

    #[test]
    fn test_bigram_forward_backward() {
        let num_buckets = 8;
        let embed_dim = 4;
        let tokens = vec![5u32, 3, 7, 1];
        let seq_len = tokens.len();

        let mut table = vec![0.0f32; num_buckets * embed_dim];
        for i in 0..table.len() {
            table[i] = (i as f32) * 0.1;
        }

        let mut output = vec![0.0f32; seq_len * embed_dim];
        bigram_hash_forward(&tokens, &table, &mut output, num_buckets, embed_dim);

        // Verify each position got the right bucket's embedding
        for t in 0..seq_len {
            let prev = if t == 0 { None } else { Some(tokens[t - 1]) };
            let bucket = bigram_hash(prev, tokens[t], num_buckets);
            for d in 0..embed_dim {
                assert_eq!(output[t * embed_dim + d], table[bucket * embed_dim + d]);
            }
        }

        // Backward: grad_output = 1.0 everywhere
        let grad_output = vec![1.0f32; seq_len * embed_dim];
        let mut grad_table = vec![0.0f32; num_buckets * embed_dim];
        bigram_hash_backward(
            &tokens,
            &grad_output,
            &mut grad_table,
            num_buckets,
            embed_dim,
        );

        let total_grad: f32 = grad_table.iter().sum();
        assert!((total_grad - (seq_len * embed_dim) as f32).abs() < 1e-6);
    }

    #[test]
    fn test_bigram_flat_position_respects_sequence_boundary() {
        let tokens = [10u32, 11, 12, 20, 21, 22];
        let num_buckets = 17;
        assert_eq!(bigram_hash_flat_position(&tokens, 0, 3, num_buckets), 16);
        assert_eq!(bigram_hash_flat_position(&tokens, 3, 3, num_buckets), 16);
        assert_eq!(
            bigram_hash_flat_position(&tokens, 4, 3, num_buckets),
            bigram_hash(Some(20), 21, num_buckets)
        );
    }

    #[test]
    fn test_embedding_bigram_project_merge_forward_matches_unfused_reference() {
        let ids = [1u32, 2, 3, 4, 5, 6];
        let seq_len = 3;
        let vocab = 8;
        let model_dim = 5;
        let bigram_vocab = 11;
        let bigram_dim = 4;
        let scale = 0.25;
        let tok_emb: Vec<f32> = (0..vocab * model_dim)
            .map(|i| (i as f32 - 7.0) * 0.03)
            .collect();
        let bigram_embed: Vec<f32> = (0..bigram_vocab * bigram_dim)
            .map(|i| (i as f32 % 13.0 - 6.0) * 0.02)
            .collect();
        let bigram_proj: Vec<f32> = (0..model_dim * bigram_dim)
            .map(|i| (i as f32 % 7.0 - 3.0) * 0.04)
            .collect();
        let mut fused = vec![0.0; ids.len() * model_dim];
        embedding_bigram_project_merge_forward(
            &ids,
            &tok_emb,
            &bigram_embed,
            &bigram_proj,
            scale,
            &mut fused,
            model_dim,
            bigram_dim,
            bigram_vocab,
            seq_len,
        );
        for t in 0..ids.len() {
            let bucket = bigram_hash_flat_position(&ids, t, seq_len, bigram_vocab);
            for j in 0..model_dim {
                let mut projected = 0.0;
                for k in 0..bigram_dim {
                    projected +=
                        bigram_embed[bucket * bigram_dim + k] * bigram_proj[j * bigram_dim + k];
                }
                let expected = tok_emb[ids[t] as usize * model_dim + j] + scale * projected;
                assert!((fused[t * model_dim + j] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_embedding_bigram_project_merge_backward_finite_difference() {
        let ids = [1u32, 2, 3, 4];
        let seq_len = 2;
        let vocab = 6;
        let model_dim = 3;
        let bigram_vocab = 7;
        let bigram_dim = 2;
        let scale = 0.4;
        let tok_emb: Vec<f32> = (0..vocab * model_dim)
            .map(|i| (i as f32 - 5.0) * 0.05)
            .collect();
        let bigram_embed: Vec<f32> = (0..bigram_vocab * bigram_dim)
            .map(|i| (i as f32 % 5.0 - 2.0) * 0.06)
            .collect();
        let bigram_proj: Vec<f32> = (0..model_dim * bigram_dim)
            .map(|i| (i as f32 % 7.0 - 3.0) * 0.03)
            .collect();
        let grad_out: Vec<f32> = (0..ids.len() * model_dim)
            .map(|i| (i as f32 % 11.0 - 4.0) * 0.07)
            .collect();
        let mut grad_tok = vec![0.0; tok_emb.len()];
        let mut grad_be = vec![0.0; bigram_embed.len()];
        let mut grad_bp = vec![0.0; bigram_proj.len()];
        let mut grad_scale = 0.0;
        embedding_bigram_project_merge_backward(
            &ids,
            &grad_out,
            &bigram_embed,
            &bigram_proj,
            scale,
            &mut grad_tok,
            &mut grad_be,
            &mut grad_bp,
            &mut grad_scale,
            model_dim,
            bigram_dim,
            bigram_vocab,
            seq_len,
        );

        fn loss(
            ids: &[u32],
            tok_emb: &[f32],
            bigram_embed: &[f32],
            bigram_proj: &[f32],
            scale: f32,
            grad_out: &[f32],
            model_dim: usize,
            bigram_dim: usize,
            bigram_vocab: usize,
            seq_len: usize,
        ) -> f32 {
            let mut out = vec![0.0; ids.len() * model_dim];
            embedding_bigram_project_merge_forward(
                ids,
                tok_emb,
                bigram_embed,
                bigram_proj,
                scale,
                &mut out,
                model_dim,
                bigram_dim,
                bigram_vocab,
                seq_len,
            );
            out.iter().zip(grad_out).map(|(a, b)| a * b).sum()
        }

        let eps = 1e-3;
        let base_args = (
            &ids,
            &tok_emb,
            &bigram_embed,
            &bigram_proj,
            scale,
            &grad_out,
            model_dim,
            bigram_dim,
            bigram_vocab,
            seq_len,
        );
        let tok_idx = ids[1] as usize * model_dim + 2;
        let mut tok_plus = tok_emb.clone();
        let mut tok_minus = tok_emb.clone();
        tok_plus[tok_idx] += eps;
        tok_minus[tok_idx] -= eps;
        let fd_tok = (loss(
            base_args.0,
            &tok_plus,
            base_args.2,
            base_args.3,
            base_args.4,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        ) - loss(
            base_args.0,
            &tok_minus,
            base_args.2,
            base_args.3,
            base_args.4,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        )) / (2.0 * eps);
        assert!((grad_tok[tok_idx] - fd_tok).abs() < 1e-4);

        let be_idx = bigram_hash_flat_position(&ids, 1, seq_len, bigram_vocab) * bigram_dim + 1;
        let mut be_plus = bigram_embed.clone();
        let mut be_minus = bigram_embed.clone();
        be_plus[be_idx] += eps;
        be_minus[be_idx] -= eps;
        let fd_be = (loss(
            base_args.0,
            base_args.1,
            &be_plus,
            base_args.3,
            base_args.4,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        ) - loss(
            base_args.0,
            base_args.1,
            &be_minus,
            base_args.3,
            base_args.4,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        )) / (2.0 * eps);
        assert!((grad_be[be_idx] - fd_be).abs() < 1e-4);

        let bp_idx = 2 * bigram_dim + 1;
        let mut bp_plus = bigram_proj.clone();
        let mut bp_minus = bigram_proj.clone();
        bp_plus[bp_idx] += eps;
        bp_minus[bp_idx] -= eps;
        let fd_bp = (loss(
            base_args.0,
            base_args.1,
            base_args.2,
            &bp_plus,
            base_args.4,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        ) - loss(
            base_args.0,
            base_args.1,
            base_args.2,
            &bp_minus,
            base_args.4,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        )) / (2.0 * eps);
        assert!((grad_bp[bp_idx] - fd_bp).abs() < 1e-4);

        let fd_scale = (loss(
            base_args.0,
            base_args.1,
            base_args.2,
            base_args.3,
            scale + eps,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        ) - loss(
            base_args.0,
            base_args.1,
            base_args.2,
            base_args.3,
            scale - eps,
            base_args.5,
            base_args.6,
            base_args.7,
            base_args.8,
            base_args.9,
        )) / (2.0 * eps);
        assert!((grad_scale - fd_scale).abs() < 1e-4);
    }

    #[test]
    fn test_bigram_merge_backward_accumulates_repeated_bucket_collisions() {
        let ids = [7u32, 7, 7, 7];
        let seq_len = 4;
        let model_dim = 3;
        let bigram_vocab = 5;
        let bigram_dim = 2;
        let bigram_embed = vec![0.2; bigram_vocab * bigram_dim];
        let bigram_proj = vec![0.3; model_dim * bigram_dim];
        let grad_out = vec![1.0; ids.len() * model_dim];
        let mut grad_tok = vec![0.0; 16 * model_dim];
        let mut grad_be = vec![0.0; bigram_vocab * bigram_dim];
        let mut grad_bp = vec![0.0; model_dim * bigram_dim];
        let mut grad_scale = 0.0;

        embedding_bigram_project_merge_backward(
            &ids,
            &grad_out,
            &bigram_embed,
            &bigram_proj,
            0.5,
            &mut grad_tok,
            &mut grad_be,
            &mut grad_bp,
            &mut grad_scale,
            model_dim,
            bigram_dim,
            bigram_vocab,
            seq_len,
        );

        assert_eq!(grad_tok[7 * model_dim..7 * model_dim + model_dim], [4.0; 3]);
        let repeated_bucket = bigram_hash(Some(7), 7, bigram_vocab);
        for k in 0..bigram_dim {
            assert!(
                (grad_be[repeated_bucket * bigram_dim + k] - 1.35).abs() < 1e-6,
                "collision bucket grad mismatch"
            );
        }
        assert!(grad_scale > 0.0);
    }

    #[test]
    fn test_fused_bigram_cuda_symbol_names_are_stable() {
        let source = include_str!("gpu_kernels.rs");
        for symbol in [
            "embedding_bigram_project_merge",
            "embedding_bigram_project_merge_backward_proj_scale",
            "embedding_bigram_project_merge_backward_embed",
        ] {
            assert!(
                source.contains(symbol),
                "CUDA BigramHash symbol {symbol} is missing"
            );
        }
    }
}
