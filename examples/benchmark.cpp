#include <catwhisper/catwhisper.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

static constexpr int N_WARMUP = 20;
static constexpr int N_BENCH  = 200;
static constexpr int RNG_SEED = 42;

struct Config {
    uint64_t n_vectors;
    uint32_t dim;
    uint32_t k;
    const char* label;
};

static const Config CONFIGS[] = {
    {  10'000, 128, 10, "10K  x128"},
    { 100'000, 128, 10, "100K x128"},
    { 100'000, 256, 10, "100K x256"},
};

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct Stats {
    double mean;
    double median;
    double p95;
    double p99;
    double qps;
};

static Stats compute_stats(std::vector<double>& ms) {
    std::sort(ms.begin(), ms.end());
    int n = static_cast<int>(ms.size());
    double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
    Stats s;
    s.mean   = sum / n;
    s.median = ms[n / 2];
    s.p95    = ms[static_cast<int>(n * 0.95)];
    s.p99    = ms[static_cast<int>(n * 0.99)];
    s.qps    = 1000.0 / s.mean;
    return s;
}

static void print_stats(const char* label, const Stats& s) {
    printf("    %-22s\n", label);
    printf("      Mean:   %8.3f ms\n",  s.mean);
    printf("      Median: %8.3f ms\n",  s.median);
    printf("      P95:    %8.3f ms\n",  s.p95);
    printf("      P99:    %8.3f ms\n",  s.p99);
    printf("      QPS:    %8.1f\n",     s.qps);
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

static std::vector<float> generate_vectors(uint64_t n, uint32_t dim, uint32_t seed = RNG_SEED) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ---------------------------------------------------------------------------
// Run one benchmark configuration
// ---------------------------------------------------------------------------

struct BenchResult {
    const char* label;
    const char* backend;
    Stats s;
};

static BenchResult run_config(cw::Context& ctx, const Config& cfg, bool use_fp16) {
    const uint64_t n = cfg.n_vectors;
    const uint32_t dim = cfg.dim;
    const uint32_t k   = cfg.k;

    printf("  Generating %llu x %u floats … ", (unsigned long long)n, dim);
    fflush(stdout);
    auto data = generate_vectors(n, dim, RNG_SEED);
    // separate seed for queries so they don't coincide with database
    auto qdata = generate_vectors(static_cast<uint64_t>(N_WARMUP + N_BENCH), dim, RNG_SEED + 1);
    printf("done\n");

    // --- create index ---
    cw::IndexOptions opts;
    opts.metric   = cw::Metric::L2;
    opts.use_fp16 = use_fp16;

    auto idx_r = cw::IndexFlat::create(ctx, dim, opts);
    if (!idx_r) {
        fprintf(stderr, "  ERROR creating index: %s\n", idx_r.error().message().c_str());
        return {};
    }
    auto& idx = *idx_r;

    // --- add vectors ---
    auto t_add0 = std::chrono::high_resolution_clock::now();
    auto add_r  = idx.add(data, n);
    auto t_add1 = std::chrono::high_resolution_clock::now();
    if (!add_r) {
        fprintf(stderr, "  ERROR adding: %s\n", add_r.error().message().c_str());
        return {};
    }
    double add_ms = std::chrono::duration<double, std::milli>(t_add1 - t_add0).count();
    printf("  [CatWhisper %s] Add: %.1f ms  (%.0f vec/s)\n",
           use_fp16 ? "fp16" : "fp32",
           add_ms, n / add_ms * 1000.0);

    // --- warmup ---
    for (int i = 0; i < N_WARMUP; ++i) {
        cw::Vector q(qdata.data() + i * dim, dim);
        auto r = idx.search(q, k);
        (void)r;
    }

    // --- timed queries ---
    std::vector<double> times_ms;
    times_ms.reserve(N_BENCH);

    for (int i = 0; i < N_BENCH; ++i) {
        cw::Vector q(qdata.data() + (N_WARMUP + i) * dim, dim);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r  = idx.search(q, k);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!r) {
            fprintf(stderr, "  ERROR searching: %s\n", r.error().message().c_str());
            return {};
        }
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    Stats s = compute_stats(times_ms);

    char backend[64];
    snprintf(backend, sizeof(backend), "catwhisper-%s-single", use_fp16 ? "fp16" : "fp32");
    print_stats(backend, s);

    BenchResult res;
    res.label   = cfg.label;
    res.backend = use_fp16 ? "catwhisper-fp16-single" : "catwhisper-fp32-single";
    res.s       = s;
    return res;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    bool large = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--large") == 0) large = true;
    }

    printf("==============================================================\n");
    printf("CatWhisper Benchmark\n");
    printf("==============================================================\n");

    auto ctx_r = cw::Context::create();
    if (!ctx_r) {
        fprintf(stderr, "Failed to create Vulkan context: %s\n",
                ctx_r.error().message().c_str());
        return 1;
    }
    auto ctx = std::move(*ctx_r);

    auto info = ctx.device_info();
    printf("Device        : %s\n", info.name.c_str());
    printf("Warmup queries: %d\n", N_WARMUP);
    printf("Bench queries : %d\n", N_BENCH);
    printf("\n");

    std::vector<BenchResult> all_results;

    // Standard configurations
    for (const auto& cfg : CONFIGS) {
        printf("--------------------------------------------------------------\n");
        printf("Config: %s  k=%u\n", cfg.label, cfg.k);

        // FP16 (default – matches library default, halves bandwidth)
        auto r16 = run_config(ctx, cfg, /*use_fp16=*/true);
        if (r16.label) all_results.push_back(r16);

        printf("\n");
    }

    // Large configuration (optional)
    if (large) {
        Config large_cfg = {1'000'000, 128, 10, "1M   x128"};
        printf("--------------------------------------------------------------\n");
        printf("Config: %s  k=%u\n", large_cfg.label, large_cfg.k);
        auto r16 = run_config(ctx, large_cfg, /*use_fp16=*/true);
        if (r16.label) all_results.push_back(r16);
        printf("\n");
    }

    // -----------------------------------------------------------------------
    // Summary table
    // -----------------------------------------------------------------------
    printf("==============================================================\n");
    printf("SUMMARY — mean search latency (ms) / QPS\n");
    printf("==============================================================\n");
    printf("%-14s %-22s %8s %8s\n", "Config", "Backend", "Mean ms", "QPS");
    printf("--------------------------------------------------------------\n");
    for (const auto& r : all_results) {
        printf("%-14s %-22s %8.3f %8.1f\n",
               r.label, r.backend, r.s.mean, r.s.qps);
    }
    printf("\n");

    // Machine-readable output for compare script
    printf("# BENCHMARK_DATA_BEGIN\n");
    for (const auto& r : all_results) {
        printf("DATA|%s|%s|%.4f|%.4f|%.4f|%.4f|%.2f\n",
               r.label, r.backend,
               r.s.mean, r.s.median, r.s.p95, r.s.p99, r.s.qps);
    }
    printf("# BENCHMARK_DATA_END\n");

    return 0;
}
