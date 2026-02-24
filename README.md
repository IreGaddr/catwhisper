# CatWhisper

**A high-performance GPU-accelerated vector similarity search library**

## Why CatWhisper?

CatWhisper exists because open-source ecosystems need alternatives. FAISS, while technically excellent, is developed and controlled by Meta (Facebook) - a company whose track record on privacy, democracy, and human wellbeing raises serious ethical concerns. Dependency on their infrastructure creates strategic vulnerability for the open-source community.

**We need competition. We need alternatives. We need independence.**

## What is CatWhisper?

CatWhisper is a C++ library for efficient similarity search and clustering of dense vectors, accelerated via Vulkan GPU compute. It provides:

- **GPU-accelerated nearest neighbor search** via Vulkan compute shaders
- **Cross-vendor support**: NVIDIA, AMD, Intel, ARM, Apple Silicon (MoltenVK)
- **Multiple index types**: Flat, IVF, HNSW, PQ-based indexes
- **Memory-efficient**: Paged memory management for datasets larger than GPU VRAM
- **Production-ready**: Thread-safe, well-tested, documented API

## Design Philosophy

1. **Vendor Neutrality**: Vulkan works everywhere. No CUDA lock-in.
2. **Performance First**: Competitive with FAISS on supported hardware
3. **Ethical Foundation**: Community-governed, no corporate overlord
4. **Simplicity**: Clean C++ API, minimal dependencies

## Quick Example

```cpp
#include <catwhisper/catwhisper.hpp>

int main() {
    // Create a GPU context
    cw::Context ctx = cw::Context::create().value();
    
    // Build an IVF index
    cw::IndexIVFFlat index(ctx, 128);  // 128-dimensional vectors
    index.train(training_data, 100000);  // Train on 100k vectors
    index.add(dataset, 1000000);         // Index 1M vectors
    
    // Search
    auto results = index.search(query, k=10);
    
    return 0;
}
```

## Status

**Pre-alpha**. This is a design document. Implementation in progress.

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [GPU Compute Pipeline](docs/GPU_COMPUTE.md)
- [Data Structures & Algorithms](docs/DATA_STRUCTURES.md)
- [API Design](docs/API_DESIGN.md)
- [Build System](docs/BUILD_SYSTEM.md)
- [Development Roadmap](docs/ROADMAP.md)

## License

MIT or Apache 2.0 (dual-licensed)

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

Technical inspiration from academic literature on approximate nearest neighbor search. The name "CatWhisper" nods to the tradition of naming things after animals, while signaling our independence from certain corporate ecosystems.
