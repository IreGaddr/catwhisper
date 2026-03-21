#ifndef CATWHISPER_CATWHISPER_HPP
#define CATWHISPER_CATWHISPER_HPP

#include <catwhisper/version.hpp>
#include <catwhisper/types.hpp>
#include <catwhisper/error.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/buffer.hpp>
#include <catwhisper/pipeline.hpp>
#include <catwhisper/index.hpp>
#include <catwhisper/index_flat.hpp>
#include <catwhisper/index_ivf_flat.hpp>
#include <catwhisper/index_ivf_pq.hpp>
#include <catwhisper/index_hnsw.hpp>
#include <catwhisper/distance.hpp>
#include <catwhisper/sparse_vector.hpp>
#include <catwhisper/distance_iot.hpp>
#include <catwhisper/index_sparse_iot.hpp>

#ifdef CW_HAS_NPU
#include <catwhisper/npu_backend.hpp>
#include <catwhisper/index_flat_npu.hpp>
#endif

namespace catwhisper = cw;

#endif // CATWHISPER_CATWHISPER_HPP
