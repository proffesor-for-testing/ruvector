//! Quantized GEMM (General Matrix Multiplication) operations.
//!
//! Core primitive for projections and FFN layers.
//! Supports int8 weights with per-row scaling.

/// Quantized GEMM: C = A * B^T + bias
///
/// Computes matrix multiplication with int8 inputs, accumulating to i64 for safety.
///
/// # Arguments
///
/// * `m` - Number of rows in A (and output C)
/// * `n` - Number of columns in B^T (and output C) = number of rows in B
/// * `k` - Number of columns in A = number of columns in B
/// * `a` - Input activations, shape [m, k], int8
/// * `a_scale` - Scale factor for input activations
/// * `b` - Weight matrix, shape [n, k], int8 (row-major, transposed)
/// * `b_row_scales` - Per-row scale factors for B, shape [n]
/// * `bias` - Optional bias vector, shape [n], i32
/// * `out` - Output buffer, shape [m, n], i32
///
/// # Output
///
/// out[i, j] = (sum_k(a[i, k] * b[j, k]) * a_scale * b_row_scales[j]) + bias[j]
///
/// # Safety
///
/// Uses i64 accumulator to prevent overflow even with large k values.
/// Bounds checking is performed at runtime for release builds.
#[inline(never)]
pub fn qgemm_i8(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    // Runtime bounds checking (critical for safety)
    if a.len() < m.saturating_mul(k)
        || b.len() < n.saturating_mul(k)
        || out.len() < m.saturating_mul(n)
        || b_row_scales.len() < n {
        // Fill with zeros on invalid dimensions rather than panicking
        for v in out.iter_mut() {
            *v = 0;
        }
        return;
    }

    // Scalar implementation with safety and scale application
    for i in 0..m {
        for j in 0..n {
            // Use i64 accumulator to prevent overflow with large k
            let mut acc: i64 = 0;

            // Dot product with bounds-checked access
            for kk in 0..k {
                let a_idx = i * k + kk;
                let b_idx = j * k + kk;

                // Safe indexing with fallback
                let a_val = a.get(a_idx).copied().unwrap_or(0) as i64;
                let b_val = b.get(b_idx).copied().unwrap_or(0) as i64;
                acc = acc.saturating_add(a_val.saturating_mul(b_val));
            }

            // Apply scale factors: acc * a_scale * b_row_scales[j]
            let combined_scale = a_scale * b_row_scales.get(j).copied().unwrap_or(1.0);
            let scaled_acc = (acc as f64 * combined_scale as f64).round() as i64;

            // Add bias if present
            let bias_val = bias.and_then(|b| b.get(j)).copied().unwrap_or(0) as i64;
            let final_acc = scaled_acc.saturating_add(bias_val);

            // Clamp to i32 range and store
            let out_idx = i * n + j;
            if let Some(out_val) = out.get_mut(out_idx) {
                *out_val = final_acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }
    }
}

/// SIMD-optimized quantized GEMM.
///
/// Uses architecture-specific SIMD when available, falls back to scalar.
#[cfg(feature = "simd")]
#[inline(never)]
pub fn qgemm_i8_simd(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    // For now, delegate to scalar. SIMD implementation would go here.
    // In production, this would use:
    // - x86_64: AVX2/AVX-512 VNNI instructions
    // - aarch64: NEON or SVE2 dot product instructions
    qgemm_i8(m, n, k, a, a_scale, b, b_row_scales, bias, out)
}

#[cfg(not(feature = "simd"))]
#[inline(never)]
pub fn qgemm_i8_simd(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    qgemm_i8(m, n, k, a, a_scale, b, b_row_scales, bias, out)
}

/// Quantized matrix-vector multiplication.
///
/// Specialized for single-row input (common in autoregressive generation).
///
/// # Safety
///
/// Uses i64 accumulator and bounds-checked access for safety.
#[inline]
pub fn qgemv_i8(
    n: usize,
    k: usize,
    x: &[i8],
    x_scale: f32,
    w: &[i8],
    w_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    // Runtime bounds checking
    if x.len() < k || w.len() < n.saturating_mul(k) || out.len() < n || w_row_scales.len() < n {
        for v in out.iter_mut() {
            *v = 0;
        }
        return;
    }

    for j in 0..n {
        // Use i64 accumulator for overflow safety
        let mut acc: i64 = 0;

        for kk in 0..k {
            let x_val = x.get(kk).copied().unwrap_or(0) as i64;
            let w_val = w.get(j * k + kk).copied().unwrap_or(0) as i64;
            acc = acc.saturating_add(x_val.saturating_mul(w_val));
        }

        // Apply scale factors
        let combined_scale = x_scale * w_row_scales.get(j).copied().unwrap_or(1.0);
        let scaled_acc = (acc as f64 * combined_scale as f64).round() as i64;

        // Add bias
        let bias_val = bias.and_then(|b| b.get(j)).copied().unwrap_or(0) as i64;
        let final_acc = scaled_acc.saturating_add(bias_val);

        // Store with clamping
        if let Some(out_val) = out.get_mut(j) {
            *out_val = final_acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
}

/// Dequantize i32 accumulator to f32.
#[inline]
pub fn dequantize_i32_to_f32(
    values: &[i32],
    input_scale: f32,
    weight_scales: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(values.len(), output.len());
    debug_assert_eq!(values.len(), weight_scales.len());

    for (i, (&v, &ws)) in values.iter().zip(weight_scales.iter()).enumerate() {
        output[i] = (v as f32) * input_scale * ws;
    }
}

/// Quantize f32 to i8 with scale.
#[inline]
pub fn quantize_f32_to_i8(values: &[f32], scale: f32, output: &mut [i8]) {
    debug_assert_eq!(values.len(), output.len());

    let inv_scale = 1.0 / scale;
    for (i, &v) in values.iter().enumerate() {
        let q = (v * inv_scale).round();
        output[i] = q.clamp(-128.0, 127.0) as i8;
    }
}

/// Compute scale factor for quantization.
#[inline]
pub fn compute_scale(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 127.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_qgemm_basic() {
        // 2x3 * 4x3^T = 2x4
        let a: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let b: [i8; 12] = [
            1, 0, 0,  // row 0
            0, 1, 0,  // row 1
            0, 0, 1,  // row 2
            1, 1, 1,  // row 3
        ];
        let scales: [f32; 4] = [1.0; 4];
        let mut out = [0i32; 8];

        qgemm_i8(2, 4, 3, &a, 1.0, &b, &scales, None, &mut out);

        // Row 0 of A: [1, 2, 3]
        // Row 0 of B: [1, 0, 0] -> dot = 1
        // Row 1 of B: [0, 1, 0] -> dot = 2
        // Row 2 of B: [0, 0, 1] -> dot = 3
        // Row 3 of B: [1, 1, 1] -> dot = 6
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 2);
        assert_eq!(out[2], 3);
        assert_eq!(out[3], 6);
    }

    #[test]
    fn test_qgemm_with_bias() {
        let a: [i8; 4] = [1, 1, 1, 1];
        let b: [i8; 4] = [1, 1, 1, 1];
        let scales: [f32; 2] = [1.0; 2];
        let bias: [i32; 2] = [10, 20];
        let mut out = [0i32; 4];

        qgemm_i8(2, 2, 2, &a, 1.0, &b, &scales, Some(&bias), &mut out);

        // Each dot product = 2, plus bias
        assert_eq!(out[0], 12); // 2 + 10
        assert_eq!(out[1], 22); // 2 + 20
    }

    #[test]
    fn test_qgemv() {
        let x: [i8; 3] = [1, 2, 3];
        let w: [i8; 6] = [
            1, 0, 0,  // row 0
            0, 1, 0,  // row 1
        ];
        let scales: [f32; 2] = [1.0; 2];
        let mut out = [0i32; 2];

        qgemv_i8(2, 3, &x, 1.0, &w, &scales, None, &mut out);

        assert_eq!(out[0], 1);
        assert_eq!(out[1], 2);
    }

    #[test]
    fn test_quantize_dequantize() {
        let original: [f32; 4] = [0.5, -0.25, 1.0, -1.0];
        let scale = compute_scale(&original);

        let mut quantized = [0i8; 4];
        quantize_f32_to_i8(&original, scale, &mut quantized);

        let scales = [scale; 4];
        let quantized_i32: Vec<i32> = quantized.iter().map(|&x| x as i32).collect();
        let mut recovered = [0.0f32; 4];
        dequantize_i32_to_f32(&quantized_i32, 1.0, &scales, &mut recovered);

        // Check approximate recovery (quantization loses precision)
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.02);
        }
    }
}
