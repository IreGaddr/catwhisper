# Data Structures & Algorithms

## Overview

CatWhisper implements several index types, each optimized for different use cases:

| Index | Use Case | Accuracy | Speed | Memory |
|-------|----------|----------|-------|--------|
| IndexFlat | Exact search, small datasets | 100% | Linear | Full |
| IndexIVFFlat | Large datasets, balanced | 95-99% | Fast | Full |
| IndexIVFPQ | Very large datasets | 90-98% | Fast | Compressed |
| IndexHNSW | Low latency, high accuracy | 95-99% | Very Fast | High |

## IndexFlat: Brute Force

The baseline index computes distances to all vectors.

### Data Layout

```
GPU Memory Layout (row-major, float16):
┌─────────────────────────────────────────────────────┐
│ v0_d0 v0_d1 v0_d2 ... v0_d127 v1_d0 v1_d1 ... vN_d127│
└─────────────────────────────────────────────────────┘

n_vectors * dimension * sizeof(float16_t)
```

### Algorithm

```
Input: query vector q, database vectors v[0..N-1], k
Output: indices and distances of k nearest neighbors

1. For each vector v[i] in parallel:
   - Compute distance d[i] = ||q - v[i]||_2
   
2. Find top-k smallest distances
   - Use GPU heap-based selection or bitonic sort

3. Return top-k indices and distances
```

### Complexity

- Build: O(n) - just copy data
- Search: O(n * d) - compute all distances
- Memory: O(n * d) - full precision storage

### When to Use

- Datasets < 1M vectors with GPU
- Exact search required
- Baseline for benchmarking

---

## IndexIVFFlat: Inverted File Index

Partitions vectors into clusters, searches only relevant clusters.

### Data Structures

```cpp
struct IVFFlatIndex {
    // Centroids: nlist * dimension
    float16_t* centroids;      // GPU memory
    
    // Inverted lists: one list per cluster
    struct InvertedList {
        uint32_t* ids;         // Vector IDs
        float16_t* vectors;    // Actual vectors
        uint32_t size;         // Number of vectors in this list
        uint32_t capacity;     // Allocated capacity
    };
    
    InvertedList* lists;       // nlist lists
    uint32_t nlist;            // Number of clusters
    
    // For GPU: contiguous storage with offsets
    uint32_t* list_offsets;    // Start offset of each list
    uint32_t* list_sizes;      // Size of each list
    float16_t* all_vectors;    // All vectors, sorted by cluster
    uint32_t* all_ids;         // All IDs
};
```

### GPU Memory Layout

```
Centroids (nlist * dimension):
┌────────────────────────────────────────┐
│ c0_d0 ... c0_dD │ c1_d0 ... c1_dD │ ... │
└────────────────────────────────────────┘

List Offsets (nlist + 1):
┌────────────────────────────────────────┐
│ 0 │ list0_size │ list0+list1_size │ ...│
└────────────────────────────────────────┘

Vectors (n * dimension):
┌─────────────────────────────────────────────────────┐
│ cluster0_vectors │ cluster1_vectors │ ...           │
└─────────────────────────────────────────────────────┘

IDs (n):
┌─────────────────────────────────────────────────────┐
│ cluster0_ids │ cluster1_ids │ ...                   │
└─────────────────────────────────────────────────────┘
```

### Training (K-means)

```cpp
void trainIVF(const float* data, uint32_t n, uint32_t nlist) {
    // Initialize centroids with k-means++
    
    // 1. Choose first centroid randomly
    centroids[0] = data[random()];
    
    // 2. Choose remaining centroids with probability proportional to D^2
    for (uint32_t c = 1; c < nlist; ++c) {
        // Compute distances to nearest centroid
        for (uint32_t i = 0; i < n; ++i) {
            float min_dist = INF;
            for (uint32_t j = 0; j < c; ++j) {
                float d = distance(data[i], centroids[j]);
                min_dist = std::min(min_dist, d);
            }
            D2[i] = min_dist * min_dist;
        }
        
        // Sample proportional to D2
        centroids[c] = data[sample_proportional(D2)];
    }
    
    // 3. Iterate k-means until convergence
    for (int iter = 0; iter < max_iters; ++iter) {
        // Assign each vector to nearest centroid
        for (uint32_t i = 0; i < n; ++i) {
            assignments[i] = argmin_j(distance(data[i], centroids[j]));
        }
        
        // Recompute centroids
        for (uint32_t j = 0; j < nlist; ++j) {
            centroids[j] = mean({data[i] : assignments[i] == j});
        }
    }
}
```

### Search Algorithm

```
Input: query q, nprobe clusters to search
Output: top-k results

1. Find nprobe nearest centroids to q
   - Compute distance to all centroids (small)
   - Top-nprobe selection

2. For each selected cluster c:
   - Compute distance from q to all vectors in inverted list[c]
   
3. Merge results from all probed clusters
   - Maintain global top-k heap

4. Return top-k indices and distances
```

### Complexity

- Train: O(n * nlist * iterations * d)
- Add: O(n * nlist * d) - find cluster for each vector
- Search: O(nprobe * avg_list_size * d)
- Memory: O(n * d + nlist * d)

### Parameters

```cpp
struct IVFParams {
    uint32_t nlist = (int)sqrt(n);  // Number of clusters
    uint32_t nprobe = 16;           // Clusters to search (tradeoff)
    
    // Higher nprobe = better accuracy, slower search
    // Typical: nprobe=1 gives ~70% recall, nprobe=64 gives ~98% recall
};
```

---

## IndexIVFPQ: Product Quantization

Compresses vectors for memory efficiency while maintaining reasonable accuracy.

### Product Quantization Concept

Split each vector into m subvectors, quantize each independently:

```
Original vector (128-dim):
┌─────────────────────────────────────────────────────┐
│ d0 ... d15 │ d16 ... d31 │ ... │ d112 ... d127     │
└─────────────────────────────────────────────────────┘
  subvec 0     subvec 1           subvec 7

Each subvector quantized to one of 256 centroids (8 bits)
```

### Data Structures

```cpp
struct IVFPQIndex {
    // Coarse quantizer (same as IVF)
    float16_t* centroids;     // nlist * dimension
    
    // Product quantizer
    uint32_t m;               // Number of subquantizers
    uint32_t dsub;            // Subvector dimension = dimension / m
    uint32_t nbits;           // Bits per subquantizer (typically 8)
    
    // PQ codebooks: m * 2^nbits * dsub
    float16_t* codebooks;     // Centroids for each subquantizer
    
    // PQ codes: n * m bytes (for 8-bit)
    uint8_t* codes;           // Quantized vectors
    
    // Inverted lists
    uint32_t* list_offsets;
    uint32_t* list_sizes;
    uint32_t* all_ids;
};
```

### Training PQ Codebooks

```cpp
void trainPQ(const float* data, uint32_t n, uint32_t m, uint32_t nbits) {
    uint32_t dsub = dimension / m;
    uint32_t k = 1 << nbits;  // 256 for 8 bits
    
    for (uint32_t sub = 0; sub < m; ++sub) {
        // Extract subvectors
        float* subvectors = extract_subvectors(data, sub, dsub);
        
        // Run k-means on subvectors
        codebooks[sub] = kmeans(subvectors, n, k);
    }
}
```

### Encoding

```cpp
void encode(const float* vector, uint8_t* code) {
    for (uint32_t sub = 0; sub < m; ++sub) {
        // Extract subvector
        float* subvec = vector + sub * dsub;
        
        // Find nearest centroid in codebook
        uint8_t best = 0;
        float best_dist = INF;
        for (uint32_t c = 0; c < 256; ++c) {
            float d = distance(subvec, codebooks[sub][c]);
            if (d < best_dist) {
                best_dist = d;
                best = c;
            }
        }
        code[sub] = best;
    }
}
```

### Asymmetric Distance Computation

Precompute distance tables for efficient search:

```cpp
// Precompute: for each subquantizer and each centroid
// dist_table[sub][c] = ||query_subvec[sub] - codebook[sub][c]||^2

void precomputeDistanceTable(const float* query, float* dist_table) {
    for (uint32_t sub = 0; sub < m; ++sub) {
        for (uint32_t c = 0; c < 256; ++c) {
            dist_table[sub * 256 + c] = 
                distance(query + sub * dsub, codebooks[sub * 256 + c]);
        }
    }
}

// Then, distance to any database vector is just table lookups
float computeDistance(const uint8_t* code, const float* dist_table) {
    float dist = 0;
    for (uint32_t sub = 0; sub < m; ++sub) {
        dist += dist_table[sub * 256 + code[sub]];
    }
    return dist;
}
```

### GPU Implementation

```glsl
// pq_distance.comp
layout(set = 0, binding = 0) readonly buffer CodeBuffer {
    uint8_t codes[];  // PQ codes: n * m
};

layout(set = 0, binding = 1) readonly buffer TableBuffer {
    float dist_table[];  // nprobe * m * 256
};

layout(set = 0, binding = 2) writeonly buffer OutputBuffer {
    float distances[];
};

void main() {
    uint vec_idx = gl_GlobalInvocationID.x;
    uint query_idx = gl_LocalInvocationID.y;
    
    float dist = 0.0;
    for (uint sub = 0; sub < m; ++sub) {
        uint code = codes[vec_idx * m + sub];
        dist += dist_table[query_idx * m * 256 + sub * 256 + code];
    }
    
    distances[vec_idx] = dist;
}
```

### Memory Savings

| Index | 1M vectors, 128-dim, float32 | Compression |
|-------|-------------------------------|-------------|
| Flat  | 512 MB | 1x |
| IVFFlat | 512 MB | 1x |
| IVFPQ (m=16) | 16 MB codes + codebook | ~30x |

---

## IndexHNSW: Hierarchical Navigable Small World

Graph-based index for very fast approximate search.

### Concept

Multi-layer graph structure:
- Layer 0: All vectors, short-range connections
- Higher layers: Subset of vectors, long-range connections
- Search starts at top layer, descends

```
Layer 2:    ●───────────────●
            │               │
Layer 1:    ●───────●───────●───────●
            │       │       │       │
Layer 0:    ●───●───●───●───●───●───●───●
            ●───●───●───●───●───●───●───●
```

### Data Structures

```cpp
struct HNSWIndex {
    // Graph structure
    struct Node {
        uint32_t id;
        uint32_t level;           // Node's max level
        float16_t* vector;        // The actual vector
    };
    
    struct Layer {
        std::vector<std::vector<uint32_t>> neighbors;  // Adjacency lists
    };
    
    std::vector<Layer> layers;
    std::vector<Node> nodes;
    
    // Parameters
    uint32_t M = 16;              // Max connections per node (layer 0: 2*M)
    uint32_t ef_construction = 200;  // Build-time search width
    double ml = 1.0 / log(M);     // Level multiplier
    
    // For GPU: CSR format
    uint32_t* edge_offsets;       // CSR offsets
    uint32_t* edges;              // CSR edges
};
```

### GPU Memory Layout (CSR)

```
Nodes (n * dimension):
┌─────────────────────────────────────────────────────┐
│ node0_vec │ node1_vec │ ... │ nodeN_vec            │
└─────────────────────────────────────────────────────┘

Node Levels (n):
┌─────────────────────────────────────────────────────┐
│ level0 │ level1 │ ... │ levelN                     │
└─────────────────────────────────────────────────────┘

Edge Offsets (n + 1):
┌─────────────────────────────────────────────────────┐
│ 0 │ deg0 │ deg0+deg1 │ ... │ total_edges           │
└─────────────────────────────────────────────────────┘

Edges (total_edges):
┌─────────────────────────────────────────────────────┐
│ neighbors of node 0 │ neighbors of node 1 │ ...     │
└─────────────────────────────────────────────────────┘
```

### Search Algorithm

```cpp
std::vector<Result> search(const float* query, uint32_t k, uint32_t ef) {
    // Start at entry point (highest layer)
    uint32_t entry = entry_point_;
    int max_layer = nodes[entry].level;
    
    // Greedy search through layers
    std::vector<uint32_t> candidates = {entry};
    
    for (int layer = max_layer; layer > 0; --layer) {
        candidates = searchLayer(query, candidates, 1, layer);
    }
    
    // Final search on layer 0 with ef candidates
    candidates = searchLayer(query, candidates, ef, 0);
    
    // Return top-k
    return topK(candidates, k);
}

std::vector<uint32_t> searchLayer(const float* query, 
                                   std::vector<uint32_t> entry_points,
                                   uint32_t ef, int layer) {
    PriorityQueue visited;     // Min-heap by distance to query
    PriorityQueue candidates;  // Max-heap by distance (extract worst)
    
    for (auto ep : entry_points) {
        float d = distance(query, nodes[ep].vector);
        visited.push({ep, d});
        candidates.push({ep, d});
    }
    
    while (!candidates.empty()) {
        auto [curr, curr_dist] = candidates.extract_min();
        auto [worst, worst_dist] = visited.peek_max();
        
        if (curr_dist > worst_dist) break;  // All remaining are worse
        
        // Explore neighbors
        for (uint32_t neighbor : layers[layer].neighbors[curr]) {
            if (!visited.contains(neighbor)) {
                float d = distance(query, nodes[neighbor].vector);
                
                if (d < worst_dist || visited.size() < ef) {
                    visited.push({neighbor, d});
                    candidates.push({neighbor, d});
                    
                    if (visited.size() > ef) {
                        visited.extract_max();
                    }
                }
            }
        }
    }
    
    return visited.extract_all();
}
```

### Construction Algorithm

```cpp
void add(const float* vector, uint32_t id) {
    // Random level for this node
    int level = random_level(ml);
    
    // Find entry point
    uint32_t entry = entry_point_;
    int max_layer = nodes[entry].level;
    
    // Search through layers above node's level
    for (int layer = max_layer; layer > level; --layer) {
        entry = searchLayerOne(vector, entry, layer);
    }
    
    // Insert at each layer from level down to 0
    for (int layer = std::min(level, max_layer); layer >= 0; --layer) {
        auto neighbors = searchLayer(vector, {entry}, ef_construction_, layer);
        
        // Select best M neighbors
        auto selected = selectNeighbors(neighbors, M);
        
        // Add bidirectional edges
        layers[layer].addEdge(id, selected);
        for (auto neighbor : selected) {
            layers[layer].addEdge(neighbor, id);
            
            // Prune if too many edges
            if (layers[layer].degree(neighbor) > M) {
                auto pruned = selectNeighbors(layers[layer].neighbors[neighbor], M);
                layers[layer].setNeighbors(neighbor, pruned);
            }
        }
    }
    
    // Update entry point if needed
    if (level > max_layer) {
        entry_point_ = id;
    }
}
```

### GPU Implementation Challenges

HNSW is inherently sequential and graph-walking, which doesn't map well to GPU parallelism. Options:

1. **Hybrid**: Build on CPU, search on CPU (HNSW doesn't benefit much from GPU)
2. **Batch search**: Search many queries in parallel on GPU
3. **Alternative**: Use IVF+PQ for GPU-accelerated search

For CatWhisper, we recommend:
- IndexHNSW: CPU-only implementation (still very fast)
- IndexIVFPQ: GPU-accelerated for batch searches

---

## Distance Metrics

### L2 (Euclidean)

```
d(x, y) = ||x - y||_2 = sqrt(sum((x_i - y_i)^2))

For nearest neighbor: can skip sqrt
```

### Inner Product (IP)

```
d(x, y) = -x · y = -sum(x_i * y_i)

Negative because we want minimum distance = maximum similarity
```

### Cosine

```
d(x, y) = 1 - (x · y) / (||x|| ||y||)

For normalized vectors, equivalent to IP
```

### GPU Implementation

```glsl
float computeDistance(float16_t a[], float16_t b[], uint32_t dim, uint32_t metric) {
    float sum = 0.0;
    
    if (metric == METRIC_L2) {
        for (uint32_t i = 0; i < dim; ++i) {
            float diff = float(a[i]) - float(b[i]);
            sum += diff * diff;
        }
        return sum;  // No sqrt for comparison
    }
    else if (metric == METRIC_IP) {
        for (uint32_t i = 0; i < dim; ++i) {
            sum += float(a[i]) * float(b[i]);
        }
        return -sum;  // Negative for min-heap compatibility
    }
    else if (metric == METRIC_COSINE) {
        float dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            dot += float(a[i]) * float(b[i]);
            norm_a += float(a[i]) * float(a[i]);
            norm_b += float(b[i]) * float(b[i]);
        }
        return 1.0 - dot / (sqrt(norm_a) * sqrt(norm_b));
    }
}
```

---

## Serialization Format

All indexes use a common header:

```cpp
struct IndexHeader {
    char magic[8] = "CATWHSPR";  // Magic number
    uint32_t version;             // Format version
    uint32_t index_type;          // Flat, IVF, PQ, HNSW
    uint32_t dimension;           // Vector dimension
    uint32_t n_vectors;           // Number of vectors
    uint32_t metric;              // Distance metric
    // Index-specific data follows
};
```

### IndexFlat Format

```
[header]
[vectors: n * d * sizeof(float16_t)]
```

### IndexIVFFlat Format

```
[header]
[nlist: 4 bytes]
[centroids: nlist * d * sizeof(float16_t)]
[list_offsets: (nlist + 1) * 4 bytes]
[vectors: n * d * sizeof(float16_t)]
[ids: n * 4 bytes]
```

### IndexIVFPQ Format

```
[header]
[nlist: 4 bytes]
[centroids: nlist * d * sizeof(float16_t)]
[m: 4 bytes]
[nbits: 4 bytes]
[codebooks: m * 2^nbits * (d/m) * sizeof(float16_t)]
[list_offsets: (nlist + 1) * 4 bytes]
[codes: n * m bytes]
[ids: n * 4 bytes]
```

### IndexHNSW Format

```
[header]
[M: 4 bytes]
[ef_construction: 4 bytes]
[ml: 8 bytes (double)]
[entry_point: 4 bytes]
[node_levels: n * 4 bytes]
[edge_offsets: (n + 1) * 4 bytes]
[edges: variable]
[vectors: n * d * sizeof(float16_t)]
```
