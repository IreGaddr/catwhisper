#ifndef CATWHISPER_DISTANCE_HPP
#define CATWHISPER_DISTANCE_HPP

#include <cstdint>
#include <span>
#include <vector>

namespace cw {

namespace distance {

float l2_sqr(std::span<const float> a, std::span<const float> b);

float inner_product(std::span<const float> a, std::span<const float> b);

float cosine_similarity(std::span<const float> a, std::span<const float> b);

float cosine_distance(std::span<const float> a, std::span<const float> b);

void normalize(std::span<float> vec);

std::vector<float> normalized(std::span<const float> vec);

} // namespace distance

void normalize_batch(std::span<float> data, uint64_t n, uint32_t dim);

} // namespace cw

#endif // CATWHISPER_DISTANCE_HPP
