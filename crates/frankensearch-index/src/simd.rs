//! Portable SIMD dot-product helpers for vector search.

use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use wide::f32x8;

/// Dot product between two f32 vectors.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f32_f32(a: &[f32], b: &[f32]) -> SearchResult<f32> {
    ensure_same_len(a.len(), b.len())?;
    Ok(dot_product_f32_f32_unchecked(a, b))
}

/// Dot product between an f16 stored vector and an f32 query vector.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f16_f32(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    ensure_same_len(stored.len(), query.len())?;

    // 2 independent accumulators to break the loop-carried dependency.
    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);

    let dim = stored.len();
    let full_pairs = dim / 16;

    for pair in 0..full_pairs {
        let base = pair * 16;
        let s0 = &stored[base..];
        let q0 = &query[base..];
        sum0 += f32x8::from([
            s0[0].to_f32(),
            s0[1].to_f32(),
            s0[2].to_f32(),
            s0[3].to_f32(),
            s0[4].to_f32(),
            s0[5].to_f32(),
            s0[6].to_f32(),
            s0[7].to_f32(),
        ]) * load_f32x8(q0);

        let s1 = &stored[base + 8..];
        let q1 = &query[base + 8..];
        sum1 += f32x8::from([
            s1[0].to_f32(),
            s1[1].to_f32(),
            s1[2].to_f32(),
            s1[3].to_f32(),
            s1[4].to_f32(),
            s1[5].to_f32(),
            s1[6].to_f32(),
            s1[7].to_f32(),
        ]) * load_f32x8(q1);
    }

    // Handle one leftover 8-element chunk if dim % 16 >= 8.
    let tail_start = full_pairs * 16;
    if tail_start + 8 <= dim {
        let s = &stored[tail_start..];
        let q = &query[tail_start..];
        sum0 += f32x8::from([
            s[0].to_f32(),
            s[1].to_f32(),
            s[2].to_f32(),
            s[3].to_f32(),
            s[4].to_f32(),
            s[5].to_f32(),
            s[6].to_f32(),
            s[7].to_f32(),
        ]) * load_f32x8(q);
    }
    let processed = if tail_start + 8 <= dim {
        tail_start + 8
    } else {
        tail_start
    };

    let mut result = (sum0 + sum1).reduce_add();
    for idx in processed..dim {
        result += stored[idx].to_f32() * query[idx];
    }
    Ok(result)
}

/// Cosine similarity helper for f16 stored vectors.
///
/// Assumes both vectors are already L2-normalized and therefore returns the
/// raw dot product value.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn cosine_similarity_f16(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    dot_product_f16_f32(stored, query)
}

/// Dot product between f16 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f16s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 2`.
pub fn dot_product_f16_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 2 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 2,
        });
    }

    // Use 4 independent accumulators to break the data-dependency chain on `sum`.
    // Each multiply-accumulate feeds a different register, allowing the CPU to
    // pipeline up to 4 FMA operations per cycle instead of stalling on one.
    // (Inspired by fff.nvim / neo_frizbee's SIMD prefilter approach.)
    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);
    let mut sum2 = f32x8::splat(0.0);
    let mut sum3 = f32x8::splat(0.0);

    // Process 32 elements (4 SIMD chunks) per iteration.
    let full_groups = dim / 32;
    for group in 0..full_groups {
        let base_byte = group * 64; // 32 elements * 2 bytes
        let base_q = group * 32;

        // Chunk 0 → sum0
        let b = &stored_bytes[base_byte..];
        let q = &query[base_q..];
        sum0 += decode_f16x8_from_bytes(b) * load_f32x8(q);

        // Chunk 1 → sum1
        let b = &stored_bytes[base_byte + 16..];
        let q = &query[base_q + 8..];
        sum1 += decode_f16x8_from_bytes(b) * load_f32x8(q);

        // Chunk 2 → sum2
        let b = &stored_bytes[base_byte + 32..];
        let q = &query[base_q + 16..];
        sum2 += decode_f16x8_from_bytes(b) * load_f32x8(q);

        // Chunk 3 → sum3
        let b = &stored_bytes[base_byte + 48..];
        let q = &query[base_q + 24..];
        sum3 += decode_f16x8_from_bytes(b) * load_f32x8(q);
    }

    // Handle remaining complete 8-element chunks (0..3 leftover).
    let tail_start = full_groups * 32;
    let remaining_chunks = (dim - tail_start) / 8;
    for chunk in 0..remaining_chunks {
        let byte_offset = (tail_start + chunk * 8) * 2;
        let query_offset = tail_start + chunk * 8;
        sum0 += decode_f16x8_from_bytes(&stored_bytes[byte_offset..])
            * load_f32x8(&query[query_offset..]);
    }

    // Reduce all 4 accumulators.
    let mut result = (sum0 + sum1 + sum2 + sum3).reduce_add();

    // Scalar remainder (< 8 elements).
    let processed = tail_start + remaining_chunks * 8;
    for index in processed..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }

    Ok(result)
}

/// Dot product between f32 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f32s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 4`.
pub fn dot_product_f32_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 4 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 4,
        });
    }

    // 4 independent accumulators — same pipeline-breaking trick as f16 path.
    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);
    let mut sum2 = f32x8::splat(0.0);
    let mut sum3 = f32x8::splat(0.0);

    let full_groups = dim / 32;
    for group in 0..full_groups {
        let base_byte = group * 128; // 32 elements * 4 bytes
        let base_q = group * 32;

        let b = &stored_bytes[base_byte..];
        let q = &query[base_q..];
        sum0 += decode_f32x8_from_bytes(b) * load_f32x8(q);

        let b = &stored_bytes[base_byte + 32..];
        let q = &query[base_q + 8..];
        sum1 += decode_f32x8_from_bytes(b) * load_f32x8(q);

        let b = &stored_bytes[base_byte + 64..];
        let q = &query[base_q + 16..];
        sum2 += decode_f32x8_from_bytes(b) * load_f32x8(q);

        let b = &stored_bytes[base_byte + 96..];
        let q = &query[base_q + 24..];
        sum3 += decode_f32x8_from_bytes(b) * load_f32x8(q);
    }

    let tail_start = full_groups * 32;
    let remaining_chunks = (dim - tail_start) / 8;
    for chunk in 0..remaining_chunks {
        let byte_offset = (tail_start + chunk * 8) * 4;
        let query_offset = tail_start + chunk * 8;
        sum0 += decode_f32x8_from_bytes(&stored_bytes[byte_offset..])
            * load_f32x8(&query[query_offset..]);
    }

    let mut result = (sum0 + sum1 + sum2 + sum3).reduce_add();

    let processed = tail_start + remaining_chunks * 8;
    for index in processed..dim {
        let b = &stored_bytes[index * 4..];
        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        result = val.mul_add(query[index], result);
    }

    Ok(result)
}

fn dot_product_f32_f32_unchecked(a: &[f32], b: &[f32]) -> f32 {
    // 2 independent accumulators to break the loop-carried dependency.
    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);

    let full_pairs = a.len() / 16; // 2 chunks of 8 per iteration
    for pair in 0..full_pairs {
        let base = pair * 16;
        sum0 += load_f32x8(&a[base..]) * load_f32x8(&b[base..]);
        sum1 += load_f32x8(&a[base + 8..]) * load_f32x8(&b[base + 8..]);
    }

    // Handle one leftover 8-element chunk if dim % 16 >= 8.
    let tail_start = full_pairs * 16;
    if tail_start + 8 <= a.len() {
        sum0 += load_f32x8(&a[tail_start..]) * load_f32x8(&b[tail_start..]);
    }
    let processed = if tail_start + 8 <= a.len() {
        tail_start + 8
    } else {
        tail_start
    };

    let mut result = (sum0 + sum1).reduce_add();
    for (x, y) in a[processed..].iter().zip(&b[processed..]) {
        result += x * y;
    }
    result
}

// ─── SIMD helper functions ─────────────────────────────────────────────────

/// Load 8 f32 values from a slice into a SIMD vector.
#[inline]
fn load_f32x8(s: &[f32]) -> f32x8 {
    f32x8::from([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

/// Decode 8 f16 values from little-endian bytes into a SIMD f32x8 vector.
#[inline]
fn decode_f16x8_from_bytes(b: &[u8]) -> f32x8 {
    f32x8::from([
        f16::from_le_bytes([b[0], b[1]]).to_f32(),
        f16::from_le_bytes([b[2], b[3]]).to_f32(),
        f16::from_le_bytes([b[4], b[5]]).to_f32(),
        f16::from_le_bytes([b[6], b[7]]).to_f32(),
        f16::from_le_bytes([b[8], b[9]]).to_f32(),
        f16::from_le_bytes([b[10], b[11]]).to_f32(),
        f16::from_le_bytes([b[12], b[13]]).to_f32(),
        f16::from_le_bytes([b[14], b[15]]).to_f32(),
    ])
}

/// Decode 8 f32 values from little-endian bytes into a SIMD f32x8 vector.
#[inline]
fn decode_f32x8_from_bytes(b: &[u8]) -> f32x8 {
    f32x8::from([
        f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        f32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        f32::from_le_bytes([b[16], b[17], b[18], b[19]]),
        f32::from_le_bytes([b[20], b[21], b[22], b[23]]),
        f32::from_le_bytes([b[24], b[25], b[26], b[27]]),
        f32::from_le_bytes([b[28], b[29], b[30], b[31]]),
    ])
}

const fn ensure_same_len(expected: usize, found: usize) -> SearchResult<()> {
    if expected != found {
        return Err(SearchError::DimensionMismatch { expected, found });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn scalar_dot_f16(stored: &[f16], query: &[f32]) -> f32 {
        stored.iter().zip(query).map(|(x, y)| x.to_f32() * y).sum()
    }

    fn normalize(vec: &[f32]) -> Vec<f32> {
        let norm = vec.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            return vec.to_vec();
        }
        vec.iter().map(|value| value / norm).collect()
    }

    #[test]
    fn simd_matches_scalar_f32() {
        let a = vec![
            0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55,
            0.31,
        ];
        let b = vec![
            -0.8, 0.7, 0.6, -0.2, 0.3, 0.9, -0.4, 0.1, 0.12, 0.21, -0.14, 0.75, -0.22, 0.35, 0.66,
            -0.19,
        ];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn simd_matches_scalar_f16() {
        let query = vec![
            0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55,
            0.31,
        ];
        let stored = vec![
            f16::from_f32(-0.8),
            f16::from_f32(0.7),
            f16::from_f32(0.6),
            f16::from_f32(-0.2),
            f16::from_f32(0.3),
            f16::from_f32(0.9),
            f16::from_f32(-0.4),
            f16::from_f32(0.1),
            f16::from_f32(0.12),
            f16::from_f32(0.21),
            f16::from_f32(-0.14),
            f16::from_f32(0.75),
            f16::from_f32(-0.22),
            f16::from_f32(0.35),
            f16::from_f32(0.66),
            f16::from_f32(-0.19),
        ];
        let simd = dot_product_f16_f32(&stored, &query).expect("dot product");
        let scalar = scalar_dot_f16(&stored, &query);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn remainder_elements_are_handled() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn zero_vector_dot_product_is_zero() {
        let stored = vec![f16::from_f32(0.0); 16];
        let query = vec![1.0; 16];
        let result = dot_product_f16_f32(&stored, &query).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn nan_input_propagates_nan() {
        let mut a = vec![1.0; 16];
        a[3] = f32::NAN;
        let b = vec![1.0; 16];
        let result = dot_product_f32_f32(&a, &b).expect("dot product");
        assert!(result.is_nan());
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let a = vec![1.0; 8];
        let b = vec![1.0; 7];
        let err = dot_product_f32_f32(&a, &b).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 7
            }
        ));
    }

    #[test]
    fn f16_precision_error_is_bounded_for_unit_vectors() {
        let pattern = [
            0.11_f32, -0.07, 0.19, 0.02, -0.13, 0.23, 0.31, -0.17, 0.05, -0.29, 0.37, 0.41,
        ];
        let mut stored_full = Vec::with_capacity(384);
        let mut query = Vec::with_capacity(384);
        for index in 0..384 {
            let value = pattern[index % pattern.len()];
            let other = pattern[(index + 3) % pattern.len()];
            stored_full.push(value);
            query.push(other);
        }
        let stored_full = normalize(&stored_full);
        let query = normalize(&query);
        let stored_f16: Vec<f16> = stored_full.iter().copied().map(f16::from_f32).collect();

        let f32_dot = scalar_dot_f32(&stored_full, &query);
        let f16_dot = dot_product_f16_f32(&stored_f16, &query).expect("dot product");
        assert!(
            (f32_dot - f16_dot).abs() < 0.01,
            "f32_dot={f32_dot}, f16_dot={f16_dot}"
        );
    }

    // ─── bd-1l4g tests begin ───

    #[test]
    fn cosine_similarity_f16_matches_dot_product() {
        let stored: Vec<f16> = (0_u16..16)
            .map(|i| f16::from_f32(f32::from(i) * 0.1))
            .collect();
        let query: Vec<f32> = (0_u16..16).map(|i| f32::from(i) * 0.2).collect();

        let cosine = cosine_similarity_f16(&stored, &query).expect("cosine");
        let dot = dot_product_f16_f32(&stored, &query).expect("dot");
        assert!(
            (cosine - dot).abs() < f32::EPSILON,
            "cosine_similarity_f16 should delegate to dot_product_f16_f32"
        );
    }

    #[test]
    fn cosine_similarity_f16_dimension_mismatch() {
        let stored = vec![f16::from_f32(1.0); 8];
        let query = vec![1.0_f32; 9];
        let err = cosine_similarity_f16(&stored, &query).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 9
            }
        ));
    }

    #[test]
    fn dot_product_f16_f32_dimension_mismatch() {
        let stored = vec![f16::from_f32(1.0); 4];
        let query = vec![1.0_f32; 5];
        let err = dot_product_f16_f32(&stored, &query).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 4,
                found: 5
            }
        ));
    }

    #[test]
    fn empty_vectors_dot_product_f32() {
        let result = dot_product_f32_f32(&[], &[]).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn empty_vectors_dot_product_f16() {
        let result = dot_product_f16_f32(&[], &[]).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn exactly_eight_elements_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!(
            (simd - scalar).abs() < 1e-6,
            "exactly 8 elements (one full SIMD chunk, no remainder)"
        );
    }

    #[test]
    fn single_element_dot_product() {
        let a = vec![3.0_f32];
        let b = vec![4.0_f32];
        let result = dot_product_f32_f32(&a, &b).expect("dot product");
        assert!((result - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn self_dot_product_is_norm_squared() {
        let v = vec![3.0_f32, 4.0];
        let result = dot_product_f32_f32(&v, &v).expect("dot product");
        assert!((result - 25.0).abs() < f32::EPSILON); // 3^2 + 4^2 = 25
    }

    #[test]
    fn f16_nan_propagates() {
        let stored = vec![
            f16::from_f32(1.0),
            f16::NAN,
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];
        let query = vec![1.0_f32; 4];
        let result = dot_product_f16_f32(&stored, &query).expect("dot product");
        assert!(result.is_nan());
    }

    #[test]
    fn large_256d_matches_scalar_f32() {
        let a: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.01).sin()).collect();
        let b: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.02).cos()).collect();
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!(
            (simd - scalar).abs() < 1e-4,
            "256d: simd={simd}, scalar={scalar}"
        );
    }

    // ─── bd-1l4g tests end ───
}
