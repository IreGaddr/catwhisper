#include "core/context_impl.hpp"
#include <catwhisper/buffer.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/index_ivf_pq.hpp>
#include <catwhisper/pipeline.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <set>

#if defined(__AVX__)
#include <immintrin.h>
#define CW_HAVE_AVX 1
#endif

namespace cw {

// Float32 to float16 conversion
static uint16_t float_to_half(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(float));

  uint32_t sign = (bits >> 31) & 1;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;

  if (exponent > 15) {
    return static_cast<uint16_t>((sign << 15) | 0x7C00);
  }
  if (exponent < -14) {
    return static_cast<uint16_t>(sign << 15);
  }

  int32_t new_exp = exponent + 15;
  uint32_t new_mant = mantissa >> 13;

  return static_cast<uint16_t>((sign << 15) | (new_exp << 10) | new_mant);
}

// K-means for IVF coarse quantizer
class IVFKMeans {
public:
  IVFKMeans(uint32_t n_clusters, uint32_t dimension, uint32_t max_iters,
            Metric metric)
      : n_clusters_(n_clusters), dimension_(dimension), max_iters_(max_iters),
        metric_(metric) {}

  std::vector<float> fit(const float *data, uint64_t n_samples) {
    if (n_samples < n_clusters_) {
      n_clusters_ = static_cast<uint32_t>(n_samples);
    }

    std::vector<float> centroids(n_clusters_ * dimension_);
    kmeans_plusplus_init(data, n_samples, centroids.data());

    std::vector<uint32_t> assignments(n_samples);

    for (uint32_t iter = 0; iter < max_iters_; ++iter) {
      assign_clusters(data, n_samples, centroids.data(), assignments.data());
      if (update_centroids(data, n_samples, assignments.data(),
                           centroids.data()))
        break;
    }

    return centroids;
  }

  uint32_t nclusters() const { return n_clusters_; }

private:
  uint32_t n_clusters_;
  uint32_t dimension_;
  uint32_t max_iters_;
  Metric metric_;

  void kmeans_plusplus_init(const float *data, uint64_t n_samples,
                            float *centroids) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> uniform_dist(0, n_samples - 1);

    uint64_t first_idx = uniform_dist(rng);
    std::memcpy(centroids, data + first_idx * dimension_,
                dimension_ * sizeof(float));

    std::vector<double> min_distances(n_samples,
                                      std::numeric_limits<double>::max());

    for (uint32_t c = 1; c < n_clusters_; ++c) {
      double total_dist = 0.0;
      for (uint64_t i = 0; i < n_samples; ++i) {
        double dist = compute_distance(data + i * dimension_,
                                       centroids + (c - 1) * dimension_);
        if (dist < min_distances[i])
          min_distances[i] = dist;
        total_dist += min_distances[i];
      }

      std::uniform_real_distribution<double> prob_dist(0.0, total_dist);
      double threshold = prob_dist(rng);
      double cumulative = 0.0;
      uint64_t next_idx = 0;
      for (uint64_t i = 0; i < n_samples; ++i) {
        cumulative += min_distances[i];
        if (cumulative >= threshold) {
          next_idx = i;
          break;
        }
      }

      std::memcpy(centroids + c * dimension_, data + next_idx * dimension_,
                  dimension_ * sizeof(float));
    }
  }

  double compute_distance(const float *a, const float *b) const {
    double dist = 0.0;
    if (metric_ == Metric::L2) {
      for (uint32_t d = 0; d < dimension_; ++d) {
        double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        dist += diff * diff;
      }
    } else {
      for (uint32_t d = 0; d < dimension_; ++d) {
        dist += static_cast<double>(a[d]) * static_cast<double>(b[d]);
      }
      dist = -dist;
    }
    return dist;
  }

  void assign_clusters(const float *data, uint64_t n_samples,
                       const float *centroids, uint32_t *assignments) {
    for (uint64_t i = 0; i < n_samples; ++i) {
      double min_dist = std::numeric_limits<double>::max();
      uint32_t best = 0;
      for (uint32_t c = 0; c < n_clusters_; ++c) {
        double dist =
            compute_distance(data + i * dimension_, centroids + c * dimension_);
        if (dist < min_dist) {
          min_dist = dist;
          best = c;
        }
      }
      assignments[i] = best;
    }
  }

  bool update_centroids(const float *data, uint64_t n_samples,
                        const uint32_t *assignments, float *centroids) {
    std::vector<double> new_centroids(n_clusters_ * dimension_, 0.0);
    std::vector<uint64_t> counts(n_clusters_, 0);

    for (uint64_t i = 0; i < n_samples; ++i) {
      uint32_t c = assignments[i];
      counts[c]++;
      for (uint32_t d = 0; d < dimension_; ++d) {
        new_centroids[c * dimension_ + d] +=
            static_cast<double>(data[i * dimension_ + d]);
      }
    }

    bool converged = true;
    for (uint32_t c = 0; c < n_clusters_; ++c) {
      if (counts[c] > 0) {
        for (uint32_t d = 0; d < dimension_; ++d) {
          float new_val =
              static_cast<float>(new_centroids[c * dimension_ + d] / counts[c]);
          if (std::abs(new_val - centroids[c * dimension_ + d]) > 1e-6f)
            converged = false;
          centroids[c * dimension_ + d] = new_val;
        }
      }
    }
    return converged;
  }
};

// PQ codebook trainer - trains m subquantizers independently
class PQTrainer {
public:
  PQTrainer(uint32_t m, uint32_t nbits, uint32_t subdim, Metric metric)
      : m_(m), nbits_(nbits), subdim_(subdim), metric_(metric) {}

  // Train codebooks: returns m * ncentroids * subdim floats
  std::vector<float> train(const float *data, uint64_t n_samples) {
    uint32_t ncentroids = 1u << nbits_; // 256 for 8-bit
    std::vector<float> codebooks(m_ * ncentroids * subdim_);

    std::mt19937 rng(42);

    for (uint32_t sub = 0; sub < m_; ++sub) {
      // Extract subvectors for this subquantizer
      std::vector<float> subvectors(n_samples * subdim_);
      for (uint64_t i = 0; i < n_samples; ++i) {
        for (uint32_t d = 0; d < subdim_; ++d) {
          subvectors[i * subdim_ + d] =
              data[i * (m_ * subdim_) + sub * subdim_ + d];
        }
      }

      // K-means on subvectors
      std::vector<float> sub_codebook =
          train_subquantizer(subvectors.data(), n_samples, ncentroids, rng);

      // Copy to output
      std::memcpy(codebooks.data() + sub * ncentroids * subdim_,
                  sub_codebook.data(), ncentroids * subdim_ * sizeof(float));
    }

    return codebooks;
  }

private:
  uint32_t m_;
  uint32_t nbits_;
  uint32_t subdim_;
  Metric metric_;

  std::vector<float> train_subquantizer(const float *data, uint64_t n_samples,
                                        uint32_t ncentroids,
                                        std::mt19937 &rng) {
    if (n_samples < ncentroids) {
      ncentroids = static_cast<uint32_t>(n_samples);
    }

    std::vector<float> centroids(ncentroids * subdim_);

    // K-means++ initialization
    std::uniform_int_distribution<uint64_t> uniform_dist(0, n_samples - 1);
    uint64_t first_idx = uniform_dist(rng);
    std::memcpy(centroids.data(), data + first_idx * subdim_,
                subdim_ * sizeof(float));

    std::vector<double> min_distances(n_samples,
                                      std::numeric_limits<double>::max());

    for (uint32_t c = 1; c < ncentroids; ++c) {
      double total_dist = 0.0;
      for (uint64_t i = 0; i < n_samples; ++i) {
        double dist = subvector_distance(data + i * subdim_,
                                         centroids.data() + (c - 1) * subdim_);
        if (dist < min_distances[i])
          min_distances[i] = dist;
        total_dist += min_distances[i];
      }

      std::uniform_real_distribution<double> prob_dist(0.0, total_dist);
      double threshold = prob_dist(rng);
      double cumulative = 0.0;
      uint64_t next_idx = 0;
      for (uint64_t i = 0; i < n_samples; ++i) {
        cumulative += min_distances[i];
        if (cumulative >= threshold) {
          next_idx = i;
          break;
        }
      }

      std::memcpy(centroids.data() + c * subdim_, data + next_idx * subdim_,
                  subdim_ * sizeof(float));
    }

    // K-means iterations
    std::vector<uint32_t> assignments(n_samples);
    for (uint32_t iter = 0; iter < 20; ++iter) {
      // Assign
      for (uint64_t i = 0; i < n_samples; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        uint32_t best = 0;
        for (uint32_t c = 0; c < ncentroids; ++c) {
          double dist = subvector_distance(data + i * subdim_,
                                           centroids.data() + c * subdim_);
          if (dist < min_dist) {
            min_dist = dist;
            best = c;
          }
        }
        assignments[i] = best;
      }

      // Update
      std::vector<double> new_centroids(ncentroids * subdim_, 0.0);
      std::vector<uint64_t> counts(ncentroids, 0);

      for (uint64_t i = 0; i < n_samples; ++i) {
        uint32_t c = assignments[i];
        counts[c]++;
        for (uint32_t d = 0; d < subdim_; ++d) {
          new_centroids[c * subdim_ + d] += data[i * subdim_ + d];
        }
      }

      for (uint32_t c = 0; c < ncentroids; ++c) {
        if (counts[c] > 0) {
          for (uint32_t d = 0; d < subdim_; ++d) {
            centroids[c * subdim_ + d] =
                static_cast<float>(new_centroids[c * subdim_ + d] / counts[c]);
          }
        }
      }
    }

    return centroids;
  }

  double subvector_distance(const float *a, const float *b) const {
    double dist = 0.0;
    if (metric_ == Metric::L2) {
      for (uint32_t d = 0; d < subdim_; ++d) {
        double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        dist += diff * diff;
      }
    } else {
      for (uint32_t d = 0; d < subdim_; ++d) {
        dist += static_cast<double>(a[d]) * static_cast<double>(b[d]);
      }
      dist = -dist;
    }
    return dist;
  }
};

struct IndexIVFPQ::Impl {
  Context *ctx = nullptr;
  uint32_t dimension = 0;
  uint64_t n_vectors = 0;
  Metric metric = Metric::L2;

  IVFPQParams params;
  uint32_t actual_nlist = 0;
  uint32_t subdim = 0; // dimension / m

  // IVF coarse quantizer
  std::vector<float> centroids; // nlist * dimension

  // PQ codebooks (shared across all clusters): m * 256 * subdim
  std::vector<float> codebooks;

  // Inverted lists with PQ codes
  std::vector<std::vector<uint8_t>> invlists_codes; // Per cluster: n * m bytes
  std::vector<std::vector<uint64_t>> invlists_ids;
  std::vector<std::vector<float>>
      invlists_original; // Original vectors per cluster for re-ranking
  std::vector<uint32_t> cluster_offsets; // nlist + 1
  std::vector<uint64_t> flat_ids;

  // Original vectors for re-ranking
  std::vector<float>
      original_vectors;       // n_vectors * dimension (cluster-major order)
  uint32_t rerank_factor = 3; // How many more candidates to fetch for re-ranking

  // GPU buffers
  Buffer centroids_buffer; // nlist * dimension * fp32
  Buffer codebooks_buffer; // m * 256 * subdim * fp32
  Buffer pq_codes_buffer;  // n_vectors * m bytes (packed)
  Buffer offsets_buffer;   // nlist + 1 uint32
  Buffer ids_buffer;       // n_vectors uint64

  // Search buffers
  Buffer query_buffer;        // dimension fp16
  Buffer distance_tables_buf; // nprobe * m * 256 fp32
  Buffer cluster_info_buffer; // nprobe * sizeof(uvec4)
  Buffer result_dists_buffer; // nprobe * k floats
  Buffer result_idxs_buffer;  // nprobe * k uints

  // Pipeline
  Pipeline search_pipeline;
  DescriptorSet search_desc_set;
  CommandBuffer search_cmd;
  bool search_cmd_valid = false;
  uint32_t search_cmd_k = 0;
  uint32_t search_cmd_nprobe = 0;

  // Assign pipeline (for add)
  Pipeline assign_pipeline;
  DescriptorSet assign_desc_set;
  Buffer assign_vectors_buf;
  Buffer assign_output_buf;

  bool is_trained = false;
  bool gpu_dirty = true;
};

IndexIVFPQ::IndexIVFPQ() = default;
IndexIVFPQ::IndexIVFPQ(IndexIVFPQ &&other) noexcept
    : impl_(std::move(other.impl_)) {}
IndexIVFPQ &IndexIVFPQ::operator=(IndexIVFPQ &&other) noexcept {
  impl_ = std::move(other.impl_);
  return *this;
}
IndexIVFPQ::~IndexIVFPQ() {
} // Empty body forces unique_ptr deleter instantiation here

Expected<IndexIVFPQ> IndexIVFPQ::create(Context &ctx, uint32_t dimension,
                                        const IVFPQParams &params,
                                        const IndexOptions &options) {
  // Validate parameters
  if (params.pq.m == 0) {
    return make_unexpected(ErrorCode::InvalidParameter, "PQ m must be > 0");
  }
  if (params.pq.nbits != 8) {
    return make_unexpected(ErrorCode::InvalidParameter,
                           "Only 8-bit PQ codes are currently supported");
  }
  if (dimension % params.pq.m != 0) {
    return make_unexpected(ErrorCode::InvalidParameter,
                           "Dimension must be divisible by m (dimension=" +
                               std::to_string(dimension) +
                               ", m=" + std::to_string(params.pq.m) + ")");
  }

  IndexIVFPQ index;
  index.impl_ = std::make_unique<Impl>();
  index.impl_->ctx = &ctx;
  index.impl_->dimension = dimension;
  index.impl_->metric = options.metric;
  index.impl_->params = params;
  index.impl_->subdim = dimension / params.pq.m;

  index.impl_->invlists_codes.resize(params.ivf.nlist);
  index.impl_->invlists_ids.resize(params.ivf.nlist);
  index.impl_->invlists_original.resize(params.ivf.nlist);
  index.impl_->cluster_offsets.resize(params.ivf.nlist + 1, 0);

  auto pipeline_result = index.init_pipelines();
  if (!pipeline_result) {
    return make_unexpected(pipeline_result.error().code(),
                           pipeline_result.error().message());
  }

  return index;
}

Expected<void> IndexIVFPQ::init_pipelines() {
  // Search pipeline
  {
    PipelineDesc desc;
    desc.shader_name = "pq_distance";
    desc.bindings = {
        {0, DescriptorBinding::StorageBuffer}, // distance_tables
        {1, DescriptorBinding::StorageBuffer}, // pq_codes
        {2, DescriptorBinding::StorageBuffer}, // cluster_info
        {3, DescriptorBinding::StorageBuffer}, // out_distances
        {4, DescriptorBinding::StorageBuffer}, // out_indices
    };
    desc.push_constant_size = 16; // m, k, metric, nprobe

    auto result = Pipeline::create(*impl_->ctx, desc);
    if (!result) {
      return make_unexpected(result.error().code(),
                             "Failed to create PQ search pipeline: " +
                                 result.error().message());
    }
    impl_->search_pipeline = std::move(*result);

    auto desc_result =
        DescriptorSet::create(*impl_->ctx, impl_->search_pipeline);
    if (!desc_result) {
      return make_unexpected(desc_result.error().code(),
                             "Failed to create PQ descriptor set");
    }
    impl_->search_desc_set = std::move(*desc_result);
  }

  // Assign pipeline (reuse from IVF)
  {
    PipelineDesc desc;
    desc.shader_name = "assign_clusters";
    desc.bindings = {
        {0, DescriptorBinding::StorageBuffer}, // centroids
        {1, DescriptorBinding::StorageBuffer}, // vectors
        {2, DescriptorBinding::StorageBuffer}, // cluster_ids output
    };
    desc.push_constant_size = 20;

    auto result = Pipeline::create(*impl_->ctx, desc);
    if (!result) {
      return make_unexpected(result.error().code(),
                             "Failed to create assign pipeline: " +
                                 result.error().message());
    }
    impl_->assign_pipeline = std::move(*result);

    auto desc_result =
        DescriptorSet::create(*impl_->ctx, impl_->assign_pipeline);
    if (!desc_result) {
      return make_unexpected(desc_result.error().code(),
                             "Failed to create assign descriptor set");
    }
    impl_->assign_desc_set = std::move(*desc_result);
  }

  return {};
}

uint32_t IndexIVFPQ::dimension() const { return impl_ ? impl_->dimension : 0; }
uint64_t IndexIVFPQ::size() const { return impl_ ? impl_->n_vectors : 0; }
bool IndexIVFPQ::is_trained() const {
  return impl_ ? impl_->is_trained : false;
}
uint32_t IndexIVFPQ::nlist() const { return impl_ ? impl_->actual_nlist : 0; }
uint32_t IndexIVFPQ::nprobe() const {
  return impl_ ? impl_->params.ivf.nprobe : 0;
}
void IndexIVFPQ::set_nprobe(uint32_t nprobe) {
  if (impl_)
    impl_->params.ivf.nprobe = std::min(nprobe, impl_->params.ivf.nlist);
}
uint32_t IndexIVFPQ::pq_m() const { return impl_ ? impl_->params.pq.m : 0; }
uint32_t IndexIVFPQ::pq_nbits() const {
  return impl_ ? impl_->params.pq.nbits : 0;
}
uint32_t IndexIVFPQ::pq_subdim() const { return impl_ ? impl_->subdim : 0; }
uint32_t IndexIVFPQ::rerank_factor() const {
  return impl_ ? impl_->rerank_factor : 1;
}
void IndexIVFPQ::set_rerank_factor(uint32_t factor) {
  if (impl_ && factor > 0)
    impl_->rerank_factor = factor;
}

IndexStats IndexIVFPQ::stats() const {
  IndexStats s{};
  if (impl_) {
    s.n_vectors = impl_->n_vectors;
    s.dimension = impl_->dimension;
    s.is_trained = impl_->is_trained;
    s.gpu_memory_used =
        impl_->pq_codes_buffer.size() + impl_->centroids_buffer.size() +
        impl_->codebooks_buffer.size() + impl_->ids_buffer.size();
  }
  return s;
}

Expected<void> IndexIVFPQ::train(std::span<const float> data, uint64_t n) {
  if (!impl_ || !impl_->ctx)
    return make_unexpected(ErrorCode::InvalidParameter,
                           "Index not initialized");

  uint64_t expected_size = n * impl_->dimension;
  if (data.size() < expected_size)
    return make_unexpected(ErrorCode::InvalidParameter, "Data size mismatch");

  // Step 1: Train IVF coarse quantizer
  IVFKMeans ivf_kmeans(impl_->params.ivf.nlist, impl_->dimension,
                       impl_->params.ivf.kmeans_iters, impl_->metric);
  impl_->centroids = ivf_kmeans.fit(data.data(), n);
  impl_->actual_nlist = ivf_kmeans.nclusters();
  impl_->params.ivf.nprobe =
      std::min(impl_->params.ivf.nprobe, impl_->actual_nlist);

  // Upload centroids to GPU
  uint64_t centroids_size =
      impl_->actual_nlist * impl_->dimension * sizeof(float);
  if (!impl_->centroids_buffer.valid() ||
      impl_->centroids_buffer.size() < centroids_size) {
    BufferDesc desc = {};
    desc.size = centroids_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::DeviceLocal;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create centroids buffer");
    impl_->centroids_buffer = std::move(*buf);
  }

  std::vector<uint8_t> centroid_bytes(centroids_size);
  std::memcpy(centroid_bytes.data(), impl_->centroids.data(), centroids_size);
  auto upload_result = impl_->centroids_buffer.upload(centroid_bytes);
  if (!upload_result)
    return upload_result;

  // Step 2: Train PQ codebooks (shared across all clusters)
  // Train on residuals
  std::vector<float> residuals(n * impl_->dimension);
  for (uint64_t i = 0; i < n; ++i) {
    const float *vec = data.data() + i * impl_->dimension;
    float min_dist = std::numeric_limits<float>::max();
    uint32_t best_c = 0;
    for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
      float dist = 0.0f;
      for (uint32_t d = 0; d < impl_->dimension; ++d) {
        float diff = vec[d] - impl_->centroids[c * impl_->dimension + d];
        dist += diff * diff;
      }
      if (dist < min_dist) {
        min_dist = dist;
        best_c = c;
      }
    }
    for (uint32_t d = 0; d < impl_->dimension; ++d) {
      residuals[i * impl_->dimension + d] =
          vec[d] - impl_->centroids[best_c * impl_->dimension + d];
    }
  }

  PQTrainer pq_trainer(impl_->params.pq.m, impl_->params.pq.nbits,
                       impl_->subdim, impl_->metric);
  impl_->codebooks = pq_trainer.train(residuals.data(), n);

  // Debug: Measure average reconstruction error
  double total_recon_error = 0.0;
  for (uint64_t i = 0; i < std::min(n, (uint64_t)1000); ++i) {
    const float *vec = data.data() + i * impl_->dimension;
    double vec_error = 0.0;
    for (uint32_t sub = 0; sub < impl_->params.pq.m; ++sub) {
      const float *subvec = vec + sub * impl_->subdim;
      const float *sub_codebook =
          impl_->codebooks.data() + sub * 256 * impl_->subdim;

      // Find nearest centroid
      float min_dist = std::numeric_limits<float>::max();
      for (uint32_t c = 0; c < 256; ++c) {
        float dist = 0.0f;
        for (uint32_t d = 0; d < impl_->subdim; ++d) {
          float diff = subvec[d] - sub_codebook[c * impl_->subdim + d];
          dist += diff * diff;
        }
        if (dist < min_dist) {
          min_dist = dist;
        }
      }
      vec_error += min_dist;
    }
    total_recon_error += vec_error;
  }

  // Debug: Show codebook centroid norms to verify they're reasonable
  for (uint32_t sub = 0; sub < std::min(4u, impl_->params.pq.m); ++sub) {
    for (uint32_t c = 0; c < 5; ++c) {
      float norm = 0.0f;
      for (uint32_t d = 0; d < impl_->subdim; ++d) {
        float val =
            impl_->codebooks[sub * 256 * impl_->subdim + c * impl_->subdim + d];
        norm += val * val;
      }
    }
  }

  // Upload codebooks to GPU
  uint64_t codebooks_size =
      impl_->params.pq.m * 256 * impl_->subdim * sizeof(float);
  if (!impl_->codebooks_buffer.valid() ||
      impl_->codebooks_buffer.size() < codebooks_size) {
    BufferDesc desc = {};
    desc.size = codebooks_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::DeviceLocal;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create codebooks buffer");
    impl_->codebooks_buffer = std::move(*buf);
  }

  std::vector<uint8_t> codebook_bytes(codebooks_size);
  std::memcpy(codebook_bytes.data(), impl_->codebooks.data(), codebooks_size);
  auto upload_cb = impl_->codebooks_buffer.upload(codebook_bytes);
  if (!upload_cb)
    return upload_cb;

  impl_->cluster_offsets.resize(impl_->actual_nlist + 1, 0);
  impl_->is_trained = true;

  return {};
}

// Encode a single vector into PQ codes
static void encode_vector(const float *vec, uint32_t dimension, uint32_t m,
                          uint32_t subdim, const float *codebooks,
                          uint8_t *codes, Metric metric) {
  for (uint32_t sub = 0; sub < m; ++sub) {
    const float *subvec = vec + sub * subdim;
    const float *sub_codebook = codebooks + sub * 256 * subdim;

    // Find nearest centroid in this subquantizer
    float min_dist = std::numeric_limits<float>::max();
    uint8_t best_code = 0;

    for (uint32_t c = 0; c < 256; ++c) {
      float dist = 0.0f;
      if (metric == Metric::L2) {
        for (uint32_t d = 0; d < subdim; ++d) {
          float diff = subvec[d] - sub_codebook[c * subdim + d];
          dist += diff * diff;
        }
      } else {
        for (uint32_t d = 0; d < subdim; ++d) {
          dist += subvec[d] * sub_codebook[c * subdim + d];
        }
        dist = -dist;
      }

      if (dist < min_dist) {
        min_dist = dist;
        best_code = static_cast<uint8_t>(c);
      }
    }

    codes[sub] = best_code;
  }
}

Expected<void> IndexIVFPQ::add(std::span<const float> data, uint64_t n,
                               std::span<const VectorId> ids) {
  if (!impl_ || !impl_->ctx)
    return make_unexpected(ErrorCode::InvalidParameter,
                           "Index not initialized");
  if (!impl_->is_trained)
    return make_unexpected(ErrorCode::InvalidParameter,
                           "Index must be trained first");

  uint64_t expected_size = n * impl_->dimension;
  if (data.size() < expected_size)
    return make_unexpected(ErrorCode::InvalidParameter, "Data size mismatch");

  if (n == 0)
    return {};

  // Allocate GPU buffers for cluster assignment
  uint64_t vectors_size = n * impl_->dimension * sizeof(uint16_t);
  if (!impl_->assign_vectors_buf.valid() ||
      impl_->assign_vectors_buf.size() < vectors_size) {
    BufferDesc desc = {};
    desc.size = vectors_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::HostCoherent;
    desc.map_on_create = true;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create assign vectors buffer");
    impl_->assign_vectors_buf = std::move(*buf);
  }

  uint64_t output_size = n * sizeof(uint32_t);
  if (!impl_->assign_output_buf.valid() ||
      impl_->assign_output_buf.size() < output_size) {
    BufferDesc desc = {};
    desc.size = output_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc;
    desc.memory_type = MemoryType::HostReadback;
    desc.map_on_create = true;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create assign output buffer");
    impl_->assign_output_buf = std::move(*buf);
  }

  // Convert vectors to fp16 and upload
  {
    uint16_t *mapped =
        reinterpret_cast<uint16_t *>(impl_->assign_vectors_buf.mapped());
    for (uint64_t i = 0; i < n * impl_->dimension; ++i) {
      mapped[i] = float_to_half(data[i]);
    }
  }

  // GPU cluster assignment
  auto bind0 = impl_->assign_desc_set.bind_buffer(0, impl_->centroids_buffer);
  auto bind1 = impl_->assign_desc_set.bind_buffer(1, impl_->assign_vectors_buf);
  auto bind2 = impl_->assign_desc_set.bind_buffer(2, impl_->assign_output_buf);

  if (!bind0 || !bind1 || !bind2) {
    return make_unexpected(ErrorCode::OperationFailed,
                           "Failed to bind assign descriptor set");
  }

  auto cmd_result = CommandBuffer::create(*impl_->ctx);
  if (!cmd_result)
    return make_unexpected(cmd_result.error().code(),
                           cmd_result.error().message());

  CommandBuffer cmd = std::move(*cmd_result);

  struct PushConstants {
    uint32_t n_clusters;
    uint32_t dimension;
    uint32_t n_vectors;
    uint32_t metric;
    uint32_t padding;
  } pc = {impl_->actual_nlist, impl_->dimension, static_cast<uint32_t>(n),
          (impl_->metric == Metric::IP) ? 1u : 0u, 0u};

  cmd.begin();
  cmd.bind_pipeline(impl_->assign_pipeline);
  cmd.bind_descriptor_set(impl_->assign_pipeline, impl_->assign_desc_set);
  cmd.push_constants(impl_->assign_pipeline, &pc, sizeof(pc));
  cmd.dispatch(static_cast<uint32_t>(n));
  cmd.end();

  auto submit_result = submit_and_wait(*impl_->ctx, cmd);
  if (!submit_result)
    return make_unexpected(submit_result.error().code(),
                           submit_result.error().message());

  // Read cluster assignments and encode vectors
  const uint32_t *cluster_ids =
      reinterpret_cast<const uint32_t *>(impl_->assign_output_buf.mapped());

  for (uint64_t i = 0; i < n; ++i) {
    uint32_t cluster = cluster_ids[i];
    if (cluster >= impl_->actual_nlist) {
      cluster = 0;
    }

    // Encode vector into PQ codes (using residual)
    const float *vec = data.data() + i * impl_->dimension;
    std::vector<float> residual(impl_->dimension);
    for (uint32_t d = 0; d < impl_->dimension; ++d) {
      residual[d] = vec[d] - impl_->centroids[cluster * impl_->dimension + d];
    }
    std::vector<uint8_t> codes(impl_->params.pq.m);
    encode_vector(residual.data(), impl_->dimension, impl_->params.pq.m,
                  impl_->subdim, impl_->codebooks.data(), codes.data(),
                  impl_->metric);

    // Append to inverted list
    impl_->invlists_codes[cluster].insert(impl_->invlists_codes[cluster].end(),
                                          codes.begin(), codes.end());

    VectorId external_id = ids.empty() ? (impl_->n_vectors + i) : ids[i];
    impl_->invlists_ids[cluster].push_back(external_id);

    // Store original vector for re-ranking (in cluster order)
    impl_->invlists_original[cluster].insert(
        impl_->invlists_original[cluster].end(), vec, vec + impl_->dimension);
  }

  impl_->n_vectors += n;
  impl_->gpu_dirty = true;

  // Debug: print cluster distribution
  if (n > 1000) {
    for (uint32_t c = 0; c < std::min(8u, impl_->actual_nlist); ++c) {
    }
  }

  return {};
}

Expected<void> IndexIVFPQ::upload_to_gpu() {
  if (!impl_->gpu_dirty || impl_->n_vectors == 0)
    return {};

  // Compute cluster offsets
  impl_->cluster_offsets[0] = 0;
  for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
    impl_->cluster_offsets[c + 1] =
        impl_->cluster_offsets[c] +
        static_cast<uint32_t>(impl_->invlists_ids[c].size());
  }

  // Build packed PQ codes buffer
  uint64_t codes_size = impl_->n_vectors * impl_->params.pq.m;
  if (!impl_->pq_codes_buffer.valid() ||
      impl_->pq_codes_buffer.size() < codes_size) {
    BufferDesc desc = {};
    desc.size = codes_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::DeviceLocal;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create PQ codes buffer");
    impl_->pq_codes_buffer = std::move(*buf);
  }

  // Flatten PQ codes in cluster-major order
  std::vector<uint8_t> flat_codes(codes_size);
  uint64_t write_pos = 0;
  for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
    uint64_t cluster_codes_size = impl_->invlists_codes[c].size();
    std::memcpy(flat_codes.data() + write_pos, impl_->invlists_codes[c].data(),
                cluster_codes_size);
    write_pos += cluster_codes_size;
  }

  // Debug: show first few PQ codes
  if (impl_->n_vectors > 100) {
    for (uint32_t v = 0; v < std::min(5u, (uint32_t)impl_->n_vectors); ++v) {
      for (uint32_t s = 0; s < std::min(4u, impl_->params.pq.m); ++s) {
      }
    }
  }

  std::vector<uint8_t> codes_bytes(flat_codes.size());
  std::memcpy(codes_bytes.data(), flat_codes.data(), codes_bytes.size());
  auto upload_codes = impl_->pq_codes_buffer.upload(codes_bytes);
  if (!upload_codes)
    return upload_codes;

  // Upload cluster offsets
  uint64_t offsets_size = (impl_->actual_nlist + 1) * sizeof(uint32_t);
  if (!impl_->offsets_buffer.valid() ||
      impl_->offsets_buffer.size() < offsets_size) {
    BufferDesc desc = {};
    desc.size = offsets_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::DeviceLocal;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create offsets buffer");
    impl_->offsets_buffer = std::move(*buf);
  }

  std::vector<uint8_t> offset_bytes(offsets_size);
  std::memcpy(offset_bytes.data(), impl_->cluster_offsets.data(), offsets_size);
  auto upload_offsets = impl_->offsets_buffer.upload(offset_bytes);
  if (!upload_offsets)
    return upload_offsets;

  // Upload IDs
  uint64_t ids_size = impl_->n_vectors * sizeof(uint64_t);
  if (!impl_->ids_buffer.valid() || impl_->ids_buffer.size() < ids_size) {
    BufferDesc desc = {};
    desc.size = ids_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::DeviceLocal;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(), "Failed to create IDs buffer");
    impl_->ids_buffer = std::move(*buf);
  }

  impl_->flat_ids.resize(impl_->n_vectors);
  impl_->original_vectors.resize(impl_->n_vectors * impl_->dimension);
  uint64_t id_offset = 0;
  uint64_t vec_offset = 0;
  for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
    for (uint64_t id : impl_->invlists_ids[c]) {
      impl_->flat_ids[id_offset++] = id;
    }
    // Copy original vectors in cluster-major order
    uint64_t cluster_vec_count = impl_->invlists_ids[c].size();
    std::memcpy(impl_->original_vectors.data() + vec_offset,
                impl_->invlists_original[c].data(),
                cluster_vec_count * impl_->dimension * sizeof(float));
    vec_offset += cluster_vec_count * impl_->dimension;
  }

  std::vector<uint8_t> id_bytes(ids_size);
  std::memcpy(id_bytes.data(), impl_->flat_ids.data(), ids_size);
  auto upload_ids = impl_->ids_buffer.upload(id_bytes);
  if (!upload_ids)
    return upload_ids;

  impl_->gpu_dirty = false;
  impl_->search_cmd_valid = false;

  return {};
}

Expected<SearchResults> IndexIVFPQ::search(Vector query, uint32_t k) {
  if (!impl_ || !impl_->ctx)
    return make_unexpected(ErrorCode::InvalidParameter,
                           "Index not initialized");
  if (query.size() != impl_->dimension)
    return make_unexpected(ErrorCode::InvalidDimension,
                           "Query dimension mismatch");
  if (impl_->n_vectors == 0)
    return SearchResults(1, k);

  auto upload_result = upload_to_gpu();
  if (!upload_result)
    return make_unexpected(upload_result.error().code(),
                           upload_result.error().message());

  uint32_t nprobe = std::min(impl_->params.ivf.nprobe, impl_->actual_nlist);

  // Max k local to shader is 32.
  // Use rerank_factor to get more candidates for re-ranking.
  uint32_t k_rerank = std::min(k * impl_->rerank_factor, 32u);

  // Step 1: CPU - Select top-nprobe centroids
  std::vector<std::pair<float, uint32_t>> centroid_dists;
  centroid_dists.reserve(impl_->actual_nlist);
  const uint32_t dim = impl_->dimension;
#if defined(CW_HAVE_AVX)
  const uint32_t dim8 = dim / 8 * 8;
  for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
    const float *centroid = impl_->centroids.data() + c * dim;
    __m256 sum = _mm256_setzero_ps();
    if (impl_->metric == Metric::L2) {
      for (uint32_t d = 0; d < dim8; d += 8) {
        __m256 q = _mm256_loadu_ps(query.data() + d);
        __m256 cent = _mm256_loadu_ps(centroid + d);
        __m256 diff = _mm256_sub_ps(q, cent);
        sum = _mm256_fmadd_ps(diff, diff, sum);
      }
    } else {
      for (uint32_t d = 0; d < dim8; d += 8) {
        __m256 q = _mm256_loadu_ps(query.data() + d);
        __m256 cent = _mm256_loadu_ps(centroid + d);
        sum = _mm256_fmadd_ps(q, cent, sum);
      }
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum);
    float dist = 0.0f;
    for (int i = 0; i < 8; ++i) dist += tmp[i];
    for (uint32_t d = dim8; d < dim; ++d) {
      if (impl_->metric == Metric::L2) {
        float diff = query[d] - centroid[d];
        dist += diff * diff;
      } else {
        dist += query[d] * centroid[d];
      }
    }
    if (impl_->metric != Metric::L2) dist = -dist;
    centroid_dists.emplace_back(dist, c);
  }
#else
  for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
    float dist = 0.0f;
    if (impl_->metric == Metric::L2) {
      for (uint32_t d = 0; d < impl_->dimension; ++d) {
        float diff = query[d] - impl_->centroids[c * impl_->dimension + d];
        dist += diff * diff;
      }
    } else {
      for (uint32_t d = 0; d < impl_->dimension; ++d) {
        dist += query[d] * impl_->centroids[c * impl_->dimension + d];
      }
      dist = -dist;
    }
    centroid_dists.emplace_back(dist, c);
  }
#endif

  std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe,
                    centroid_dists.end());

  std::vector<uint32_t> selected_clusters;
  selected_clusters.reserve(nprobe);
  for (uint32_t p = 0; p < nprobe; ++p) {
    selected_clusters.push_back(centroid_dists[p].second);
  }

  // Step 2: Build distance tables (ADC)
  // For each selected cluster and each subquantizer, compute distance from
  // query subvector to all 256 centroids
  uint64_t tables_size = nprobe * impl_->params.pq.m * 256 * sizeof(float);
  if (!impl_->distance_tables_buf.valid() ||
      impl_->distance_tables_buf.size() < tables_size) {
    BufferDesc desc = {};
    desc.size = tables_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::HostCoherent;
    desc.map_on_create = true;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create distance tables buffer");
    impl_->distance_tables_buf = std::move(*buf);
  }

  {
    float *tables =
        reinterpret_cast<float *>(impl_->distance_tables_buf.mapped());
    for (uint32_t p = 0; p < nprobe; ++p) {
      uint32_t cluster = selected_clusters[p];
      std::vector<float> query_residual(impl_->dimension);
      float base_dist = 0.0f;
      for (uint32_t d = 0; d < impl_->dimension; ++d) {
        query_residual[d] =
            query[d] - impl_->centroids[cluster * impl_->dimension + d];
      }
      if (impl_->metric == Metric::IP) {
        float qc = 0.0f;
        for (uint32_t d = 0; d < impl_->dimension; ++d) {
          qc += query[d] * impl_->centroids[cluster * impl_->dimension + d];
        }
        base_dist = -qc;
      }

      for (uint32_t sub = 0; sub < impl_->params.pq.m; ++sub) {
        const float *query_subvec =
            (impl_->metric == Metric::L2)
                ? query_residual.data() + sub * impl_->subdim
                : query.data() + sub * impl_->subdim;
        const float *sub_codebook =
            impl_->codebooks.data() + sub * 256 * impl_->subdim;

        for (uint32_t c = 0; c < 256; ++c) {
          float dist = 0.0f;
          if (impl_->metric == Metric::L2) {
            for (uint32_t d = 0; d < impl_->subdim; ++d) {
              float diff =
                  query_subvec[d] - sub_codebook[c * impl_->subdim + d];
              dist += diff * diff;
            }
          } else {
            for (uint32_t d = 0; d < impl_->subdim; ++d) {
              dist += query_subvec[d] * sub_codebook[c * impl_->subdim + d];
            }
            dist = -dist;
            if (sub == 0) {
              dist += base_dist;
            }
          }

          // tables[p][sub][c]
          tables[p * impl_->params.pq.m * 256 + sub * 256 + c] = dist;
        }
      }
    }
    // Debug: show first few distance table entries
    static int dt_count = 0;
    if (dt_count < 3) {
      dt_count++;
    }
  }

  // Step 3: Prepare cluster info buffer
  uint64_t cluster_info_size = nprobe * sizeof(uint32_t) * 4;
  if (!impl_->cluster_info_buffer.valid() ||
      impl_->cluster_info_buffer.size() < cluster_info_size) {
    BufferDesc desc = {};
    desc.size = cluster_info_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
    desc.memory_type = MemoryType::HostCoherent;
    desc.map_on_create = true;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create cluster info buffer");
    impl_->cluster_info_buffer = std::move(*buf);
  }

  {
    uint32_t *mapped =
        reinterpret_cast<uint32_t *>(impl_->cluster_info_buffer.mapped());
    for (uint32_t p = 0; p < nprobe; ++p) {
      uint32_t cluster = selected_clusters[p];
      mapped[p * 4 + 0] =
          impl_->cluster_offsets[cluster]; // Start vector index in pq_codes
      mapped[p * 4 + 1] = impl_->cluster_offsets[cluster + 1] -
                          impl_->cluster_offsets[cluster]; // size
      mapped[p * 4 + 2] = p; // output_slot (maps to distance table)
      mapped[p * 4 + 3] = cluster;
    }
  }

  // Step 4: Allocate result buffers
  uint64_t result_size = nprobe * k_rerank * sizeof(float);
  if (!impl_->result_dists_buffer.valid() ||
      impl_->result_dists_buffer.size() < result_size) {
    BufferDesc desc = {};
    desc.size = result_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc;
    desc.memory_type = MemoryType::HostReadback;
    desc.map_on_create = true;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create result distances buffer");
    impl_->result_dists_buffer = std::move(*buf);
  }

  result_size = nprobe * k_rerank * sizeof(uint32_t);
  if (!impl_->result_idxs_buffer.valid() ||
      impl_->result_idxs_buffer.size() < result_size) {
    BufferDesc desc = {};
    desc.size = result_size;
    desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc;
    desc.memory_type = MemoryType::HostReadback;
    desc.map_on_create = true;

    auto buf = Buffer::create(*impl_->ctx, desc);
    if (!buf)
      return make_unexpected(buf.error().code(),
                             "Failed to create result indices buffer");
    impl_->result_idxs_buffer = std::move(*buf);
  }

  // Bind descriptor set
  auto bind0 =
      impl_->search_desc_set.bind_buffer(0, impl_->distance_tables_buf);
  auto bind1 = impl_->search_desc_set.bind_buffer(1, impl_->pq_codes_buffer);
  auto bind2 =
      impl_->search_desc_set.bind_buffer(2, impl_->cluster_info_buffer);
  auto bind3 =
      impl_->search_desc_set.bind_buffer(3, impl_->result_dists_buffer);
  auto bind4 = impl_->search_desc_set.bind_buffer(4, impl_->result_idxs_buffer);

  if (!bind0 || !bind1 || !bind2 || !bind3 || !bind4) {
    return make_unexpected(ErrorCode::OperationFailed,
                           "Failed to bind descriptor set");
  }

  // Record or reuse command buffer
  bool need_rerecord = !impl_->search_cmd_valid ||
                       impl_->search_cmd_k != k_rerank ||
                       impl_->search_cmd_nprobe != nprobe;

  if (need_rerecord) {
    if (!impl_->search_cmd.valid()) {
      auto cmd_result = CommandBuffer::create(*impl_->ctx);
      if (!cmd_result)
        return make_unexpected(cmd_result.error().code(),
                               cmd_result.error().message());
      impl_->search_cmd = std::move(*cmd_result);
    } else {
      impl_->search_cmd.reset();
    }

    struct PushConstants {
      uint32_t m;
      uint32_t k;
      uint32_t metric;
      uint32_t nprobe;
    } pc = {impl_->params.pq.m, k_rerank,
            (impl_->metric == Metric::IP) ? 1u : 0u, nprobe};

    impl_->search_cmd.begin_reusable();
    impl_->search_cmd.bind_pipeline(impl_->search_pipeline);
    impl_->search_cmd.bind_descriptor_set(impl_->search_pipeline,
                                          impl_->search_desc_set);
    impl_->search_cmd.push_constants(impl_->search_pipeline, &pc, sizeof(pc));
    impl_->search_cmd.dispatch(nprobe);
    impl_->search_cmd.end();

    impl_->search_cmd_valid = true;
    impl_->search_cmd_k = k_rerank;
    impl_->search_cmd_nprobe = nprobe;
  }

  auto submit_result = submit_and_wait(*impl_->ctx, impl_->search_cmd);
  if (!submit_result)
    return make_unexpected(submit_result.error().code(),
                           submit_result.error().message());

  // Merge results
  const float *dists =
      reinterpret_cast<const float *>(impl_->result_dists_buffer.mapped());
  const uint32_t *idxs =
      reinterpret_cast<const uint32_t *>(impl_->result_idxs_buffer.mapped());

  std::vector<std::pair<float, uint32_t>> all_results;
  all_results.reserve(nprobe * k_rerank);

  for (uint32_t p = 0; p < nprobe; ++p) {
    uint32_t cluster = selected_clusters[p];
    uint32_t cluster_start = impl_->cluster_offsets[cluster];
    uint32_t cluster_size =
        impl_->cluster_offsets[cluster + 1] - impl_->cluster_offsets[cluster];
    for (uint32_t i = 0; i < k_rerank; ++i) {
      uint32_t local_pos = idxs[p * k_rerank + i];
      if (local_pos != 0xFFFFFFFFu && local_pos < cluster_size) {
        uint32_t global_idx = cluster_start + local_pos;
        all_results.emplace_back(dists[p * k_rerank + i], global_idx);
      }
    }
  }

  if (!impl_->original_vectors.empty() && impl_->metric == Metric::L2) {
    const uint32_t dim = impl_->dimension;
#if defined(CW_HAVE_AVX)
    const uint32_t dim8 = dim / 8 * 8;
    for (auto &result : all_results) {
      uint32_t global_idx = result.second;
      const float *vec =
          impl_->original_vectors.data() + global_idx * dim;
      __m256 sum = _mm256_setzero_ps();
      for (uint32_t d = 0; d < dim8; d += 8) {
        __m256 q = _mm256_loadu_ps(query.data() + d);
        __m256 v = _mm256_loadu_ps(vec + d);
        __m256 diff = _mm256_sub_ps(q, v);
        sum = _mm256_fmadd_ps(diff, diff, sum);
      }
      float true_dist = 0.0f;
      alignas(32) float tmp[8];
      _mm256_store_ps(tmp, sum);
      for (int i = 0; i < 8; ++i) true_dist += tmp[i];
      for (uint32_t d = dim8; d < dim; ++d) {
        float diff = query[d] - vec[d];
        true_dist += diff * diff;
      }
      result.first = true_dist;
    }
#else
    for (auto &result : all_results) {
      uint32_t global_idx = result.second;
      const float *vec =
          impl_->original_vectors.data() + global_idx * impl_->dimension;
      float true_dist = 0.0f;
      for (uint32_t d = 0; d < impl_->dimension; ++d) {
        float diff = query[d] - vec[d];
        true_dist += diff * diff;
      }
      result.first = true_dist;
    }
#endif
  }

  std::partial_sort(all_results.begin(),
                    all_results.begin() +
                        std::min(k, static_cast<uint32_t>(all_results.size())),
                    all_results.end());
  all_results.resize(std::min(k, static_cast<uint32_t>(all_results.size())));

  SearchResults results(1, k);
  for (size_t i = 0; i < all_results.size(); ++i) {
    results.results[i].distance = all_results[i].first;
    results.results[i].id = impl_->flat_ids[all_results[i].second];
  }

  return results;
}

Expected<SearchResults> IndexIVFPQ::search(std::span<const float> queries,
                                           uint64_t n_queries, uint32_t k) {
  if (!impl_ || !impl_->ctx)
    return make_unexpected(ErrorCode::InvalidParameter,
                           "Index not initialized");
  if (impl_->n_vectors == 0)
    return SearchResults(n_queries, k);

  SearchResults results(n_queries, k);
  for (uint64_t q = 0; q < n_queries; ++q) {
    std::vector<float> query(queries.begin() + q * impl_->dimension,
                             queries.begin() + (q + 1) * impl_->dimension);
    auto single = search(query, k);
    if (!single)
      return single;
    for (uint32_t i = 0; i < k; ++i) {
      results.results[q * k + i] = single->results[i];
    }
  }
  return results;
}

Expected<void> IndexIVFPQ::save(const std::filesystem::path &path) const {
  (void)path;
  return make_unexpected(ErrorCode::OperationFailed, "Not implemented");
}

Expected<void> IndexIVFPQ::load(const std::filesystem::path &path) {
  (void)path;
  return make_unexpected(ErrorCode::OperationFailed, "Not implemented");
}

void IndexIVFPQ::reset() {
  if (impl_) {
    impl_->n_vectors = 0;
    impl_->gpu_dirty = true;
    impl_->search_cmd_valid = false;
    for (auto &list : impl_->invlists_codes)
      list.clear();
    for (auto &list : impl_->invlists_ids)
      list.clear();
    for (auto &list : impl_->invlists_original)
      list.clear();
    impl_->original_vectors.clear();
    std::fill(impl_->cluster_offsets.begin(), impl_->cluster_offsets.end(), 0);
  }
}

} // namespace cw
