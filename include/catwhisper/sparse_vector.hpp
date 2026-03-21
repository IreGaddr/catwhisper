#ifndef CATWHISPER_SPARSE_VECTOR_HPP
#define CATWHISPER_SPARSE_VECTOR_HPP

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace cw {

// Sparse int16 vector: sorted (index, value) pairs.
// Dimension is implicit — determined by max index + 1.
// Memory: 6 bytes per non-zero element (4 byte index + 2 byte value).
struct SparseEntry {
    uint32_t index;
    int16_t  value;
};

class SparseVector {
public:
    SparseVector() = default;

    explicit SparseVector(std::vector<SparseEntry> entries)
        : entries_(std::move(entries))
    {
        std::sort(entries_.begin(), entries_.end(),
                  [](const SparseEntry& a, const SparseEntry& b) {
                      return a.index < b.index;
                  });
    }

    // Build from dense int16 span, keeping only non-zero entries.
    static SparseVector from_dense(std::span<const int16_t> dense) {
        std::vector<SparseEntry> entries;
        entries.reserve(dense.size() / 4);  // heuristic: expect ~25% non-zero
        for (uint32_t i = 0; i < dense.size(); ++i) {
            if (dense[i] != 0) {
                entries.push_back({i, dense[i]});
            }
        }
        return SparseVector(std::move(entries));
    }

    // Append a new dimension (must be > max existing index).
    void append(uint32_t index, int16_t value) {
        if (value != 0) {
            entries_.push_back({index, value});
        }
    }

    // Set or update a dimension. Maintains sort order.
    void set(uint32_t index, int16_t value) {
        auto it = std::lower_bound(entries_.begin(), entries_.end(), index,
                                   [](const SparseEntry& e, uint32_t idx) {
                                       return e.index < idx;
                                   });
        if (it != entries_.end() && it->index == index) {
            if (value == 0) {
                entries_.erase(it);
            } else {
                it->value = value;
            }
        } else if (value != 0) {
            entries_.insert(it, {index, value});
        }
    }

    // Get value at dimension (0 if absent).
    int16_t get(uint32_t index) const {
        auto it = std::lower_bound(entries_.begin(), entries_.end(), index,
                                   [](const SparseEntry& e, uint32_t idx) {
                                       return e.index < idx;
                                   });
        if (it != entries_.end() && it->index == index) {
            return it->value;
        }
        return 0;
    }

    // Sparse dot product: O(nnz_a + nnz_b) merge.
    int64_t dot(const SparseVector& other) const {
        int64_t sum = 0;
        size_t i = 0, j = 0;
        while (i < entries_.size() && j < other.entries_.size()) {
            if (entries_[i].index < other.entries_[j].index) {
                ++i;
            } else if (entries_[i].index > other.entries_[j].index) {
                ++j;
            } else {
                sum += static_cast<int64_t>(entries_[i].value) *
                       static_cast<int64_t>(other.entries_[j].value);
                ++i;
                ++j;
            }
        }
        return sum;
    }

    int64_t norm_sq() const {
        int64_t sum = 0;
        for (const auto& e : entries_) {
            sum += static_cast<int64_t>(e.value) * static_cast<int64_t>(e.value);
        }
        return sum;
    }

    uint32_t nnz() const { return static_cast<uint32_t>(entries_.size()); }

    // Effective dimensionality (max index + 1, or 0 if empty).
    uint32_t effective_dim() const {
        return entries_.empty() ? 0 : entries_.back().index + 1;
    }

    std::span<const SparseEntry> entries() const { return entries_; }
    std::vector<SparseEntry>& entries_mut() { return entries_; }
    const std::vector<SparseEntry>& entries_ref() const { return entries_; }

    bool empty() const { return entries_.empty(); }

private:
    std::vector<SparseEntry> entries_;
};

} // namespace cw

#endif // CATWHISPER_SPARSE_VECTOR_HPP
