#include <catwhisper/distance.hpp>
#include <cmath>

namespace cw {
namespace distance {

float l2_sqr(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float inner_product(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float cosine_similarity(std::span<const float> a, std::span<const float> b) {
    float dot = inner_product(a, b);
    float norm_a_sq = l2_sqr(a, std::span<float>{});
    float norm_b_sq = l2_sqr(b, std::span<float>{});
    
    if (norm_a_sq == 0 || norm_b_sq == 0) {
        return 0.0f;
    }
    
    return dot / (std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq));
}

float cosine_distance(std::span<const float> a, std::span<const float> b) {
    return 1.0f - cosine_similarity(a, b);
}

void normalize(std::span<float> vec) {
    float norm_sq = 0.0f;
    for (auto& v : vec) {
        norm_sq += v * v;
    }
    
    if (norm_sq == 0) return;
    
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (auto& v : vec) {
        v *= inv_norm;
    }
}

std::vector<float> normalized(std::span<const float> vec) {
    std::vector<float> result(vec.begin(), vec.end());
    normalize(result);
    return result;
}

} // namespace distance

void normalize_batch(std::span<float> data, uint64_t n, uint32_t dim) {
    for (uint64_t i = 0; i < n; ++i) {
        distance::normalize(data.subspan(i * dim, dim));
    }
}

} // namespace cw
