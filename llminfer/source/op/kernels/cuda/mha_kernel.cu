#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
namespace kernel {
// 每个 block 中都会调用 softmax
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  // x 是 q@k 之后的 attention score
  // x 的 shape 为 (1, 1+cache_len)
  int tid = threadIdx.x; // 如果 block 有线程 0，1，2..., 31
  int step = blockDim.x; // step = 32

  // find max value (for numerical stability)
  float max_val = tid < size ? x[tid] : 0;
  // 每个线程求的是局部位置的局部最大值
  // 比如线程 0，求的是 0，32，64... 等位置的局部最大值
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // 再将线程块的所有线程的局部最大值进行规约
  // BlockReduceMax
  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  // 得到一个 block 中最大值也就是一个 score head 中的最大值
  // 一个 (1, 1+cache_len)下的最大值
  max_val = shared_val;

  // 求 exp 的局部和
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // 求 exp 的全局和
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

__global__ void multi_head_attention_kernel(
    int32_t pos, // cache_len + 1
    int32_t seq_len,
    float* query,
    float* score_ptr,
    float* output,
    float* key_cache, // (bs, 1+cache_len, heads, head_dim)
    float* value_cache, // (bs,heads,cache_len+1,head_dim)
    int32_t kv_dim, // heads * head_dim
    int32_t kv_mul, // kv_mul == 1 时， head_offset = head * head_size, grouped
                    // query 场景时不为1
    int32_t head_num,
    int32_t head_size,
    int32_t layer_offset) {
  int head = blockIdx.x; // 一个 block 负责一个 q head 的计算，grid 是一维的

  // 依次取出 head_size 维度的 key 值
  // q_head matmul key_head = scores = (1, head_size) @ (head_size, cache_len+1)
  // = (1, cache_len+1)
  if (head >= head_num) {
    return;
  }
  // 对于每个 block, query_head = (1, head_size)
  float* query_head =
      query + head * head_size; // (1, 1, num_heads, head_dim) -> (head_dim)
  float* score_head = score_ptr +
      head * seq_len; // attention score before softmax, (cache_len+1)
  float scale = 1.f / sqrtf(head_size);

  int32_t head_offset = (head / kv_mul) * head_size;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    // 外层锁定 pos
    // +128 每个 block 一次性处理 128 个 token 的 key
    // 因为一整个 kv_cache 是所有 tansformer 块共享的，所以需要通过 layeroffset
    // 先点定位当前块的 kv_cache 的起始地址
    float* key_head =
        key_cache + layer_offset + t * kv_dim + head_offset; // (head_size)

    float score = 0.0f;
#pragma unroll
    for (int i = 0; i < head_size; i += 4) {
      // 内层锁定 head_size
      // query @ key 逐个头相乘
      // (1, head_size) @ (head_size, 1) -> (1, 1)
      float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
      float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
      if (i < head_size) {
        score += key_head_float4.x * query_head_float4.x;
      }
      if (i + 1 < head_size) {
        score += key_head_float4.y * query_head_float4.y;
      }
      if (i + 2 < head_size) {
        score += key_head_float4.z * query_head_float4.z;
      }
      if (i + 3 < head_size) {
        score += key_head_float4.w * query_head_float4.w;
      }
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();
  // 将 (1, cache_len+1) 维度的 scores 矩阵和 (cache_len+1, head_size) 维度的
  // value 矩阵进行矩阵相乘 将最终结果放入到 output_head 中
  float* output_head = output + head * head_size; // (head_size)
  // layer_offset 定位到当前 value_cache 的起始位置
  head_offset = layer_offset + (head / kv_mul) * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    // 外层锁定 head_size
    float value = 0.0f;
#pragma unroll
    for (int t = 0; t <= pos; t++) {
      // 内层锁定 cache_len+1 个单词
      // value shape (1, heads, cache_len+1，head_size)
      // values_head (cache_len+1, head_size)
      // 这个 for 循环遍历 cache_len+1
      // kv_dim == head_size
      float* value_head = value_cache + head_offset + t * kv_dim; // (head_size)
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(
    int32_t pos,
    int32_t head_num,
    int32_t layer_index,
    int32_t seq_len,
    int32_t kv_dim,
    int32_t kv_mul,
    int32_t head_size,
    const tensor::Tensor& mha_out,
    const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor,
    const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor,
    base::DeviceType device_type,
    CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  int32_t thread_num = 128;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  // 一个 block 负责一个 head 的计算
  multi_head_attention_kernel<<<head_num, thread_num, 0, stream>>>(
      pos,
      seq_len,
      query,
      score,
      output,
      key_cache,
      value_cache,
      kv_dim,
      kv_mul,
      head_num,
      head_size,
      layer_offset);
}

} // namespace kernel