#include <catwhisper/distance_iot.hpp>
#include <algorithm>
#include <cmath>

namespace cw {
namespace distance {

namespace {

constexpr double kPi = 3.14159265358979323846;
// int16 range [-32767, 32767] maps to [-pi, pi]
constexpr double kAngleScale = kPi / 32767.0;

// Per-dimension IOT distance computation.
// a_val, b_val are int16 angular positions.
// Returns the squared per-dimension distance contribution.
inline double per_dim_iot_distance_sq(int16_t a_val, int16_t b_val,
                                      double R, double r,
                                      double alpha, double beta,
                                      bool& chose_involuted) {
    // Angular delta on the torus
    const double delta = static_cast<double>(a_val - b_val) * kAngleScale;

    // Direct path: metric tensor g_uu = (R + r)^2 at outer edge (v=0 approximation)
    // This is the volume-rich region where most points live.
    const double outer_metric = (R + r) * (R + r);
    const double d_direct_sq = outer_metric * delta * delta;

    // Involuted path: maps through inner edge, metric g_uu = (R - r)^2
    // The involute shifts by pi, so the effective delta changes.
    const double inv_delta = kPi - std::abs(delta);
    const double inner_metric = (R - r) * (R - r);
    const double d_inv_raw = std::sqrt(inner_metric * inv_delta * inv_delta);
    const double d_inv = alpha * d_inv_raw + beta;
    const double d_inv_sq = d_inv * d_inv;

    chose_involuted = (d_inv_sq < d_direct_sq);
    return chose_involuted ? d_inv_sq : d_direct_sq;
}

} // namespace

float iot_distance(const SparseVector& a, const SparseVector& b,
                   const IOTParams& params) {
    // Compute effective dimension from union of active indices
    const uint32_t eff_dim = std::max(a.effective_dim(), b.effective_dim());

    // Adaptive R:r ratio based on effective dimensionality
    const double kappa = params.adaptive_kappa(eff_dim);
    const double R = kappa;  // Normalize: r = 1
    const double r = 1.0;

    double total_sq = 0.0;

    // Merge walk over both sparse vectors
    auto a_entries = a.entries();
    auto b_entries = b.entries();
    size_t ai = 0, bi = 0;

    while (ai < a_entries.size() && bi < b_entries.size()) {
        bool dummy;
        if (a_entries[ai].index < b_entries[bi].index) {
            // Only a has this dimension — b is at origin (0)
            total_sq += per_dim_iot_distance_sq(
                a_entries[ai].value, 0, R, r, params.alpha, params.beta, dummy);
            ++ai;
        } else if (a_entries[ai].index > b_entries[bi].index) {
            // Only b has this dimension
            total_sq += per_dim_iot_distance_sq(
                0, b_entries[bi].value, R, r, params.alpha, params.beta, dummy);
            ++bi;
        } else {
            // Both have this dimension
            total_sq += per_dim_iot_distance_sq(
                a_entries[ai].value, b_entries[bi].value,
                R, r, params.alpha, params.beta, dummy);
            ++ai;
            ++bi;
        }
    }

    // Remaining entries in a (b is zero)
    while (ai < a_entries.size()) {
        bool dummy;
        total_sq += per_dim_iot_distance_sq(
            a_entries[ai].value, 0, R, r, params.alpha, params.beta, dummy);
        ++ai;
    }

    // Remaining entries in b (a is zero)
    while (bi < b_entries.size()) {
        bool dummy;
        total_sq += per_dim_iot_distance_sq(
            0, b_entries[bi].value, R, r, params.alpha, params.beta, dummy);
        ++bi;
    }

    return static_cast<float>(std::sqrt(total_sq));
}

PathSignature compute_path_signature(const SparseVector& query,
                                     const SparseVector& target,
                                     const IOTParams& params) {
    const uint32_t eff_dim = std::max(query.effective_dim(), target.effective_dim());
    const double kappa = params.adaptive_kappa(eff_dim);
    const double R = kappa;
    const double r = 1.0;

    PathSignature sig;

    // Collect active dimensions (union of both vectors' indices)
    auto q_entries = query.entries();
    auto t_entries = target.entries();
    size_t qi = 0, ti = 0;

    // First pass: collect active dims and path choices
    std::vector<std::pair<uint32_t, bool>> dim_choices;
    dim_choices.reserve(q_entries.size() + t_entries.size());

    while (qi < q_entries.size() && ti < t_entries.size()) {
        bool chose_inv;
        if (q_entries[qi].index < t_entries[ti].index) {
            per_dim_iot_distance_sq(q_entries[qi].value, 0, R, r,
                                    params.alpha, params.beta, chose_inv);
            dim_choices.push_back({q_entries[qi].index, chose_inv});
            ++qi;
        } else if (q_entries[qi].index > t_entries[ti].index) {
            per_dim_iot_distance_sq(0, t_entries[ti].value, R, r,
                                    params.alpha, params.beta, chose_inv);
            dim_choices.push_back({t_entries[ti].index, chose_inv});
            ++ti;
        } else {
            per_dim_iot_distance_sq(q_entries[qi].value, t_entries[ti].value,
                                    R, r, params.alpha, params.beta, chose_inv);
            dim_choices.push_back({q_entries[qi].index, chose_inv});
            ++qi;
            ++ti;
        }
    }
    while (qi < q_entries.size()) {
        bool chose_inv;
        per_dim_iot_distance_sq(q_entries[qi].value, 0, R, r,
                                params.alpha, params.beta, chose_inv);
        dim_choices.push_back({q_entries[qi].index, chose_inv});
        ++qi;
    }
    while (ti < t_entries.size()) {
        bool chose_inv;
        per_dim_iot_distance_sq(0, t_entries[ti].value, R, r,
                                params.alpha, params.beta, chose_inv);
        dim_choices.push_back({t_entries[ti].index, chose_inv});
        ++ti;
    }

    sig.n_active = static_cast<uint32_t>(dim_choices.size());
    sig.active_dims.reserve(sig.n_active);
    sig.bits.resize((sig.n_active + 63) / 64, 0);

    for (uint32_t i = 0; i < sig.n_active; ++i) {
        sig.active_dims.push_back(dim_choices[i].first);
        if (dim_choices[i].second) {
            sig.set(i);
        }
    }

    return sig;
}

uint32_t PathSignature::hamming(const PathSignature& other) const {
    uint32_t dist = 0;
    const size_t min_words = std::min(bits.size(), other.bits.size());

    for (size_t i = 0; i < min_words; ++i) {
        dist += static_cast<uint32_t>(__builtin_popcountll(bits[i] ^ other.bits[i]));
    }
    for (size_t i = min_words; i < bits.size(); ++i) {
        dist += static_cast<uint32_t>(__builtin_popcountll(bits[i]));
    }
    for (size_t i = min_words; i < other.bits.size(); ++i) {
        dist += static_cast<uint32_t>(__builtin_popcountll(other.bits[i]));
    }

    return dist;
}

} // namespace distance
} // namespace cw
