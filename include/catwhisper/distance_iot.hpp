#ifndef CATWHISPER_DISTANCE_IOT_HPP
#define CATWHISPER_DISTANCE_IOT_HPP

#include <catwhisper/sparse_vector.hpp>
#include <cmath>
#include <cstdint>

namespace cw {

// IOT (Involuted Oblate Toroid) distance for sparse integer vectors.
//
// Per-dimension distance uses the toroidal metric with involution:
//   d_i = min(d_direct_i, alpha * d_involuted_i + beta)
//
// The binary path choice (direct vs involuted) per dimension is the
// fractal switch that defeats the curse of dimensionality.
//
// For sparse vectors, dimensions where both vectors are zero contribute
// nothing (the points are at the same position on that torus copy).
// Dimensions where only one vector is non-zero use the non-zero value
// as the angular displacement.

struct IOTParams {
    double major_radius = 30.0;    // R
    double minor_radius = 1.0;     // r
    double alpha = 0.92;           // Involuted path scale
    double beta  = 0.04;           // Involuted path offset
    double c_adaptive = 2.0;       // Adaptive scaling multiplier

    // Compute R/r for a given effective dimension using adaptive formula.
    // R(d) = d / (c * ln(d)) * r, ensuring we stay at c× critical threshold.
    double adaptive_kappa(uint32_t effective_dim) const {
        if (effective_dim < 4) return major_radius / minor_radius;
        const double d = static_cast<double>(effective_dim);
        return d / (c_adaptive * std::log(d));
    }
};

namespace distance {

// IOT distance between two sparse int16 vectors.
// Returns float distance suitable for ANN search ordering.
//
// The per-dimension computation:
//   - Treat int16 values as angular positions scaled to [-pi, pi]
//   - Compute toroidal metric: g_uu = (R + r*cos(v))^2
//   - Take min of direct and involuted path
//   - Sum across all active dimensions
//
// For the sparse index, we use a simplified but exact-ordering-preserving
// version: the int16 values represent the angular delta directly, and the
// binary path choice is determined by whether the involuted path is shorter.
float iot_distance(const SparseVector& a, const SparseVector& b,
                   const IOTParams& params = {});

// Compute the binary path signature for a pair of sparse vectors.
// Returns a bitset indicating which dimensions chose the involuted path.
// This signature IS the fractal structure used for index lookup.
//
// Output: packed bits, one per active dimension (union of both vectors' indices).
// Also returns the list of active dimension indices for unpacking.
struct PathSignature {
    std::vector<uint64_t> bits;       // Packed 64-bit words
    std::vector<uint32_t> active_dims; // Which dimensions are active
    uint32_t n_active = 0;

    bool get(uint32_t local_idx) const {
        return (bits[local_idx / 64] >> (local_idx % 64)) & 1ULL;
    }

    void set(uint32_t local_idx) {
        bits[local_idx / 64] |= (1ULL << (local_idx % 64));
    }

    // Hamming distance between two signatures (count of differing bits).
    uint32_t hamming(const PathSignature& other) const;
};

PathSignature compute_path_signature(const SparseVector& query,
                                     const SparseVector& target,
                                     const IOTParams& params = {});

} // namespace distance

} // namespace cw

#endif // CATWHISPER_DISTANCE_IOT_HPP
