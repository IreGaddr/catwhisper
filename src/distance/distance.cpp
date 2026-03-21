#include <catwhisper/distance.hpp>
#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX512BF16__)
#include <immintrin.h>
#define CW_HAVE_AVX512_BF16 1
#endif

namespace cw {
namespace distance {

#if defined(__AVX512F__)
static inline float avx512_reduce_add_ps(__m512 vec) {
    __m256 low = _mm512_castps512_ps256(vec);
    __m256 high = _mm512_extractf32x8_ps(vec, 1);
    __m256 sum = _mm256_add_ps(low, high);
    __m128 low128 = _mm256_castps256_ps128(sum);
    __m128 high128 = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    return _mm_cvtss_f32(_mm_add_ss(sum64, _mm_movehdup_ps(sum64)));
}
#endif

float l2_sqr(std::span<const float> a, std::span<const float> b) {
    const size_t n = std::min(a.size(), b.size());
    
#if defined(__AVX512F__)
    // AVX512 implementation: process 16 floats per iteration
    const size_t simd_len = n & ~15u;
    __m512 sum_vec = _mm512_setzero_ps();
    
    for (size_t i = 0; i < simd_len; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(a.data() + i);
        __m512 b_vec = _mm512_loadu_ps(b.data() + i);
        __m512 diff = _mm512_sub_ps(a_vec, b_vec);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }
    
    float sum = avx512_reduce_add_ps(sum_vec);
    
    // Handle remaining elements
    for (size_t i = simd_len; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#endif
}

float inner_product(std::span<const float> a, std::span<const float> b) {
    const size_t n = std::min(a.size(), b.size());

#if defined(__AVX512F__)
    // AVX512 FMA: sum += a * b in single fused multiply-add
    const size_t simd_len = n & ~15u;
    __m512 sum_vec = _mm512_setzero_ps();

    for (size_t i = 0; i < simd_len; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(a.data() + i);
        __m512 b_vec = _mm512_loadu_ps(b.data() + i);
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    
    float sum = avx512_reduce_add_ps(sum_vec);
    
    // Handle remaining elements
    for (size_t i = simd_len; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

float cosine_similarity(std::span<const float> a, std::span<const float> b) {
    const size_t n = std::min(a.size(), b.size());
    if (n == 0) {
        return 0.0f;
    }

#if defined(__AVX512F__)
    // Triple-accumulator: dot, norm_a, norm_b in one pass over the data
    const size_t simd_len = n & ~15u;
    __m512 dot_vec    = _mm512_setzero_ps();
    __m512 norm_a_vec = _mm512_setzero_ps();
    __m512 norm_b_vec = _mm512_setzero_ps();

    for (size_t i = 0; i < simd_len; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(a.data() + i);
        __m512 b_vec = _mm512_loadu_ps(b.data() + i);
        dot_vec    = _mm512_fmadd_ps(a_vec, b_vec, dot_vec);
        norm_a_vec = _mm512_fmadd_ps(a_vec, a_vec, norm_a_vec);
        norm_b_vec = _mm512_fmadd_ps(b_vec, b_vec, norm_b_vec);
    }

    float dot       = avx512_reduce_add_ps(dot_vec);
    float norm_a_sq = avx512_reduce_add_ps(norm_a_vec);
    float norm_b_sq = avx512_reduce_add_ps(norm_b_vec);

    // Scalar tail
    for (size_t i = simd_len; i < n; ++i) {
        dot       += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
#else
    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
#endif

    if (norm_a_sq <= 0.0f || norm_b_sq <= 0.0f) {
        return 0.0f;
    }

    return dot / (std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq));
}

float cosine_distance(std::span<const float> a, std::span<const float> b) {
    return 1.0f - cosine_similarity(a, b);
}

void normalize(std::span<float> vec) {
    const size_t n = vec.size();
    
#if defined(__AVX512F__)
    // Compute norm using AVX512
    const size_t simd_len = n & ~15u;
    __m512 norm_sq_vec = _mm512_setzero_ps();
    
    for (size_t i = 0; i < simd_len; i += 16) {
        __m512 v = _mm512_loadu_ps(vec.data() + i);
        norm_sq_vec = _mm512_fmadd_ps(v, v, norm_sq_vec);
    }
    
    float norm_sq = avx512_reduce_add_ps(norm_sq_vec);
    for (size_t i = simd_len; i < n; ++i) {
        float v = vec[i];
        norm_sq += v * v;
    }
    
    if (norm_sq == 0.0f) return;
    
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    __m512 inv_norm_vec = _mm512_set1_ps(inv_norm);
    
    // Multiply by inverse norm in AVX512
    for (size_t i = 0; i < simd_len; i += 16) {
        __m512 v = _mm512_loadu_ps(vec.data() + i);
        v = _mm512_mul_ps(v, inv_norm_vec);
        _mm512_storeu_ps(vec.data() + i, v);
    }
    
    // Handle remaining
    for (size_t i = simd_len; i < n; ++i) {
        vec[i] *= inv_norm;
    }
#else
    float norm_sq = 0.0f;
    for (auto& v : vec) {
        norm_sq += v * v;
    }
    
    if (norm_sq == 0.0f) return;
    
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (auto& v : vec) {
        v *= inv_norm;
    }
#endif
}

std::vector<float> normalized(std::span<const float> vec) {
    std::vector<float> result(vec.begin(), vec.end());
    normalize(result);
    return result;
}

} // namespace distance

void normalize_batch(std::span<float> data, uint64_t n, uint32_t dim) {
    float* base = data.data();
    #pragma omp parallel for schedule(static) if(n > 64)
    for (uint64_t i = 0; i < n; ++i) {
        distance::normalize(std::span<float>(base + i * dim, dim));
    }
}

} // namespace cw
