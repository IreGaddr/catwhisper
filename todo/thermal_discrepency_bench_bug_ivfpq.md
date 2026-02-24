# Thermal Discrepancy Bug: IVF-PQ Benchmark Latency Variance

**Status:** Open  
**Severity:** Medium (test flakiness, not correctness)  
**Component:** `tests/rotrta/pq_benchmark.cpp`  
**Affected Tests:** `RotrtaPQBenchmark.RecallAndLatency_100Kx128`, `RotrtaPQBenchmark.RecallAndLatency_10Kx128`

## Summary

The IVF-PQ ROTRTA benchmarks exhibit high latency variance (2-3×) due to GPU thermal throttling. Tests pass when GPU is cool but fail when running after other GPU-intensive tests.

## Observed Behavior

### Latency Variance

| Test Run Condition | 10K×128 Median | 100K×128 Median | Pass/Fail |
|-------------------|----------------|-----------------|-----------|
| Cold GPU (first run) | 0.58 ms | 0.90 ms | ✅ Pass |
| Warm GPU (after IVF tests) | 1.5-1.8 ms | 2.2-3.0 ms | ❌ Fail |
| After 5s cooldown | 0.60 ms | 1.1 ms | ✅ Pass |

### Budget Thresholds

| Test | Latency Budget | Cold Performance | Warm Performance |
|------|---------------|------------------|------------------|
| 10K×128 | <1.0 ms | 0.58 ms ✅ | 1.5+ ms ❌ |
| 100K×128 | <2.0 ms | 0.90 ms ✅ | 2.2+ ms ❌ |

## Root Cause Analysis

1. **GPU Thermal State:** The ROTRTA test suite runs IVF-Flat benchmarks before IVF-PQ, heating the GPU.
2. **No Thermal Headroom:** The latency budgets are tight (~40% margin when cold), insufficient for thermal variance.
3. **Hardware-Specific:** RTX 4080 Laptop GPU has limited thermal capacity in laptop form factor.

## Reproduction

```bash
# Cold run - passes
cd /home/ire/code/catwhisper/build_release
./tests/rotrta_tests --gtest_filter=*RecallAndLatency*  # Often passes first time

# Warm run - may fail
./tests/rotrta_tests  # Full suite, PQ tests run after IVF tests
./tests/rotrta_tests --gtest_filter=*100K*  # May fail if GPU still warm
```

## Potential Fixes


**Pros:** Simple, reflects real-world thermal conditions  
**Cons:** Less aggressive ROTRTA assertions

### Option A: Test Isolation
Run PQ benchmarks in separate test suite or with forced GPU cooldown.

**Pros:** Keeps tight budgets  
**Cons:** Longer test time, doesn't reflect real usage

### Option B: GPU Thermal Management
Add GPU cooling period between test suites (e.g., `sleep 5` or GPU idle loop).

**Pros:** Keeps tight budgets  
**Cons:** Slower CI, platform-dependent

### Option C: Optimize Further
Reduce CPU work in hot path to increase thermal margin.

**Status:** AVX SIMD already applied, ~30% improvement achieved. Further gains would require GPU-only pipeline.

## Current Workaround

Run PQ tests individually or after a cooldown period:

```bash
# Reliable pass
sleep 5 && ./tests/rotrta_tests --gtest_filter=*RecallAndLatency*
```

## Metrics (Pre-SIMD vs Post-SIMD)

| Metric | Before AVX | After AVX | Improvement |
|--------|-----------|-----------|-------------|
| 100K×128 cold | 1.5 ms | 0.90 ms | 40% faster |
| 100K×128 warm | 3.0 ms | 2.2 ms | 27% faster |

## Notes

- This is a **test infrastructure** issue, not a correctness bug
- Recall is stable at 77-97% regardless of thermal state
- Production usage would have more consistent thermal conditions
- Consider documenting expected performance range rather than single budget

## History

- **2026-02-24:** Issue identified during ROTRTA benchmark work
- **2026-02-24:** AVX SIMD optimization applied, reduced but not eliminated variance
