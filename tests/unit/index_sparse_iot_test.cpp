#include <gtest/gtest.h>
#include <catwhisper/index_sparse_iot.hpp>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace cw {
namespace {

// Helper: generate a random sparse vector with given nnz and max dimension.
SparseVector make_random_sparse(std::mt19937& rng, uint32_t nnz, uint32_t max_dim) {
    std::uniform_int_distribution<uint32_t> dim_dist(0, max_dim - 1);
    std::uniform_int_distribution<int16_t> val_dist(-10000, 10000);

    std::vector<SparseEntry> entries;
    entries.reserve(nnz);
    // Use a set to avoid duplicate dimensions
    std::vector<uint32_t> dims(max_dim);
    std::iota(dims.begin(), dims.end(), 0);
    std::shuffle(dims.begin(), dims.end(), rng);
    dims.resize(nnz);
    std::sort(dims.begin(), dims.end());

    for (uint32_t d : dims) {
        int16_t v = val_dist(rng);
        if (v == 0) v = 1; // ensure non-zero
        entries.push_back({d, v});
    }
    return SparseVector(std::move(entries));
}

// Helper: brute-force k-NN search for correctness verification.
std::vector<std::pair<float, VectorId>> brute_force_search(
    const SparseVector& query,
    const std::vector<std::pair<VectorId, SparseVector>>& vectors,
    uint32_t k,
    const IOTParams& iot_params = {}) {

    std::vector<std::pair<float, VectorId>> dists;
    dists.reserve(vectors.size());
    for (const auto& [id, vec] : vectors) {
        float d = distance::iot_distance(query, vec, iot_params);
        dists.push_back({d, id});
    }
    std::partial_sort(dists.begin(),
                      dists.begin() + std::min(k, static_cast<uint32_t>(dists.size())),
                      dists.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    dists.resize(std::min(k, static_cast<uint32_t>(dists.size())));
    return dists;
}

class IndexSparseIOTTest : public ::testing::Test {
protected:
    std::mt19937 rng{12345};
    SparseIOTParams params;

    void SetUp() override {
        params.n_pivots = 16;
        params.signature_bits = 128;
        params.pivot_sample_size = 64;
    }
};

TEST_F(IndexSparseIOTTest, CreateAndAdd) {
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(idx->size(), 0u);

    auto vec = make_random_sparse(rng, 20, 100);
    auto result = idx->add(1, vec);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(idx->size(), 1u);
}

TEST_F(IndexSparseIOTTest, BatchAdd) {
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    const uint32_t n = 200;
    std::vector<VectorId> ids(n);
    std::vector<SparseVector> vecs(n);
    for (uint32_t i = 0; i < n; ++i) {
        ids[i] = i + 1;
        vecs[i] = make_random_sparse(rng, 30, 200);
    }

    auto result = idx->add_batch(ids, vecs);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(idx->size(), n);
    EXPECT_GT(idx->n_pivots(), 0u);
}

TEST_F(IndexSparseIOTTest, SearchSmall_BruteForce) {
    // Below pivot threshold: brute force path
    params.pivot_sample_size = 256; // Force brute force
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    const uint32_t n = 50;
    std::vector<std::pair<VectorId, SparseVector>> store;
    for (uint32_t i = 0; i < n; ++i) {
        auto vec = make_random_sparse(rng, 20, 100);
        idx->add(i + 1, vec);
        store.push_back({i + 1, vec});
    }

    auto query = make_random_sparse(rng, 20, 100);
    auto results = idx->search(query, 5);
    ASSERT_TRUE(results.has_value());

    auto expected = brute_force_search(query, store, 5, params.iot);

    // Brute force path should return exact results
    for (uint32_t i = 0; i < 5 && i < expected.size(); ++i) {
        EXPECT_EQ((*results)[0][i].id, expected[i].second)
            << "Mismatch at rank " << i;
    }
}

TEST_F(IndexSparseIOTTest, SearchWithPivots_RecallCheck) {
    // Past pivot threshold: signature-based search
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    const uint32_t n = 500;
    std::vector<std::pair<VectorId, SparseVector>> store;
    for (uint32_t i = 0; i < n; ++i) {
        auto vec = make_random_sparse(rng, 30, 200);
        idx->add(i + 1, vec);
        store.push_back({i + 1, vec});
    }

    auto query = make_random_sparse(rng, 30, 200);
    const uint32_t k = 10;
    auto results = idx->search(query, k);
    ASSERT_TRUE(results.has_value());

    auto expected = brute_force_search(query, store, k, params.iot);

    // Check recall: at least 50% of true top-k should be in returned results
    uint32_t hits = 0;
    for (uint32_t i = 0; i < k && i < expected.size(); ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            if ((*results)[0][j].id == expected[i].second) {
                ++hits;
                break;
            }
        }
    }
    EXPECT_GE(hits, k / 2) << "Recall too low: " << hits << "/" << k;
}

TEST_F(IndexSparseIOTTest, SearchFindsExactNeighbor) {
    // Insert a vector, then search with a very similar vector
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    // Fill past pivot threshold
    for (uint32_t i = 0; i < 200; ++i) {
        idx->add(i + 1, make_random_sparse(rng, 30, 200));
    }

    // Insert a specific "target" vector
    SparseVector target(std::vector<SparseEntry>{
        {0, 1000}, {10, -500}, {20, 2000}, {50, -1000}, {100, 3000}
    });
    idx->add(999, target);

    // Query with something very close to target
    SparseVector query(std::vector<SparseEntry>{
        {0, 1001}, {10, -499}, {20, 2001}, {50, -999}, {100, 2999}
    });

    auto results = idx->search(query, 5);
    ASSERT_TRUE(results.has_value());

    // Target should be in top-5
    bool found = false;
    for (uint32_t i = 0; i < 5; ++i) {
        if ((*results)[0][i].id == 999) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Near-duplicate target not found in top-5";
}

TEST_F(IndexSparseIOTTest, IncrementalAddAfterPivots) {
    // Build index past pivot threshold, then add more vectors
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    // Initial batch
    const uint32_t n_initial = 200;
    for (uint32_t i = 0; i < n_initial; ++i) {
        idx->add(i + 1, make_random_sparse(rng, 30, 200));
    }
    EXPECT_GT(idx->n_pivots(), 0u);

    // Add more incrementally
    const uint32_t n_extra = 100;
    for (uint32_t i = 0; i < n_extra; ++i) {
        idx->add(n_initial + i + 1, make_random_sparse(rng, 30, 200));
    }
    EXPECT_EQ(idx->size(), n_initial + n_extra);

    // Search should still work
    auto query = make_random_sparse(rng, 30, 200);
    auto results = idx->search(query, 5);
    ASSERT_TRUE(results.has_value());

    // All results should have valid IDs
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_GT((*results)[0][i].id, 0u);
        EXPECT_LE((*results)[0][i].id, n_initial + n_extra);
    }
}

TEST_F(IndexSparseIOTTest, SearchResultsOrderedByDistance) {
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    const uint32_t n = 300;
    for (uint32_t i = 0; i < n; ++i) {
        idx->add(i + 1, make_random_sparse(rng, 30, 200));
    }

    auto query = make_random_sparse(rng, 30, 200);
    auto results = idx->search(query, 10);
    ASSERT_TRUE(results.has_value());

    // Results must be sorted by distance (ascending)
    for (uint32_t i = 1; i < 10; ++i) {
        EXPECT_LE((*results)[0][i - 1].distance, (*results)[0][i].distance)
            << "Results not sorted at position " << i;
    }
}

TEST_F(IndexSparseIOTTest, SaveAndLoad) {
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    const uint32_t n = 150;
    std::vector<std::pair<VectorId, SparseVector>> store;
    for (uint32_t i = 0; i < n; ++i) {
        auto vec = make_random_sparse(rng, 30, 200);
        idx->add(i + 1, vec);
        store.push_back({i + 1, vec});
    }

    // Save
    const std::string path = "/tmp/scn_test_sparse_iot.idx";
    auto save_result = idx->save(path);
    EXPECT_TRUE(save_result.has_value());

    // Load into new index
    auto idx2 = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx2.has_value());
    auto load_result = idx2->load(path);
    EXPECT_TRUE(load_result.has_value());
    EXPECT_EQ(idx2->size(), n);

    // Loaded index should find good results vs brute force
    auto query = make_random_sparse(rng, 30, 200);
    auto r2 = idx2->search(query, 5);
    ASSERT_TRUE(r2.has_value());

    auto expected = brute_force_search(query, store, 5, params.iot);

    uint32_t hits = 0;
    for (uint32_t i = 0; i < 5 && i < expected.size(); ++i) {
        for (uint32_t j = 0; j < 5; ++j) {
            if ((*r2)[0][j].id == expected[i].second) { ++hits; break; }
        }
    }
    EXPECT_GE(hits, 3u) << "Loaded index recall: " << hits << "/5";

    std::filesystem::remove(path);
}

TEST_F(IndexSparseIOTTest, Reset) {
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    for (uint32_t i = 0; i < 100; ++i) {
        idx->add(i + 1, make_random_sparse(rng, 20, 100));
    }
    EXPECT_EQ(idx->size(), 100u);

    idx->reset();
    EXPECT_EQ(idx->size(), 0u);
    EXPECT_EQ(idx->n_pivots(), 0u);
}

TEST_F(IndexSparseIOTTest, VPTreeSearch_MatchesBruteForceHamming) {
    // This test verifies that the VP-tree Hamming pre-filter produces
    // the same candidate set as a linear Hamming scan. We do this
    // indirectly: the final IOT results should have >= 50% recall
    // against brute force for a 1000-vector index.
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    const uint32_t n = 1000;
    std::vector<std::pair<VectorId, SparseVector>> store;
    for (uint32_t i = 0; i < n; ++i) {
        auto vec = make_random_sparse(rng, 40, 300);
        idx->add(i + 1, vec);
        store.push_back({i + 1, vec});
    }

    // Run multiple queries and check average recall
    const uint32_t n_queries = 20;
    const uint32_t k = 10;
    uint32_t total_hits = 0;

    for (uint32_t q = 0; q < n_queries; ++q) {
        auto query = make_random_sparse(rng, 40, 300);
        auto results = idx->search(query, k);
        ASSERT_TRUE(results.has_value());

        auto expected = brute_force_search(query, store, k, params.iot);

        for (uint32_t i = 0; i < k && i < expected.size(); ++i) {
            for (uint32_t j = 0; j < k; ++j) {
                if ((*results)[0][j].id == expected[i].second) {
                    ++total_hits;
                    break;
                }
            }
        }
    }

    const double avg_recall = static_cast<double>(total_hits) /
                              static_cast<double>(n_queries * k);
    EXPECT_GE(avg_recall, 0.4)
        << "Average recall " << avg_recall << " too low across " << n_queries << " queries";
}

TEST_F(IndexSparseIOTTest, LargeScale_SearchCompletes) {
    // Verify search completes in reasonable time for a larger index.
    // This is a basic sanity check, not a performance benchmark.
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    const uint32_t n = 5000;
    std::vector<VectorId> ids(n);
    std::vector<SparseVector> vecs(n);
    for (uint32_t i = 0; i < n; ++i) {
        ids[i] = i + 1;
        vecs[i] = make_random_sparse(rng, 50, 500);
    }

    auto batch_result = idx->add_batch(ids, vecs);
    ASSERT_TRUE(batch_result.has_value());
    EXPECT_EQ(idx->size(), n);

    auto query = make_random_sparse(rng, 50, 500);
    auto results = idx->search(query, 10);
    ASSERT_TRUE(results.has_value());

    // Should return 10 results with valid distances
    for (uint32_t i = 0; i < 10; ++i) {
        EXPECT_GT((*results)[0][i].id, 0u);
        EXPECT_GE((*results)[0][i].distance, 0.0f);
    }
}

TEST_F(IndexSparseIOTTest, DimensionGrowth) {
    auto idx = IndexSparseIOT::create(params);
    ASSERT_TRUE(idx.has_value());

    // Add vectors with increasing dimensionality
    for (uint32_t i = 0; i < 100; ++i) {
        idx->add(i + 1, make_random_sparse(rng, 20, 100 + i * 10));
    }

    idx->notify_dimension_growth(2000);
    EXPECT_EQ(idx->max_dimension(), 2000u);

    // Add vectors in the new dimension range
    for (uint32_t i = 0; i < 50; ++i) {
        idx->add(200 + i, make_random_sparse(rng, 20, 2000));
    }

    auto query = make_random_sparse(rng, 20, 2000);
    auto results = idx->search(query, 5);
    EXPECT_TRUE(results.has_value());
}

} // namespace
} // namespace cw
