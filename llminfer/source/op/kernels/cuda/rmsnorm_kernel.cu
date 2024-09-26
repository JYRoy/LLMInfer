#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

namespace kernel {

static __global__ void row_rmsnorm_f32_warp_reduce(
    float* input,
    float* weight,
    float* output,
    int size,
    float eps) {
  const int tid = threadIdx.x;
  const int lane_id = tid % warpSize;
  float square_sum = 0.0f;
  for (int i = lane_id; i < size; i += warpSize) {
    square_sum += input[i] * input[i];
  }

  using WarpReduce = cub::WarpReduce<float, 32>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ float shared_val;
  square_sum = WarpReduce(temp).Reduce(square_sum, cub::Sum());
  if (threadIdx.x == 0) {
    shared_val = square_sum;
  }
  __syncthreads();
  square_sum = shared_val;

  const float scale = rsqrtf(square_sum / static_cast<float>(size) + eps);
  for (int i = lane_id; i < size; i += warpSize) {
    output[i] = scale * input[i] * weight[i];
  }
}

static __global__ void row_rmsnorm_f32_block_reduce(
    float* input,
    float* weight,
    float* output,
    int size,
    float eps) {
  const int tid = threadIdx.x;

  float sum = 0.0f;
  // tid = 0, in[0] + in[128] + in[256] ...
  for (int i = tid; i < size; i += blockDim.x) {
    sum = input[i] * input[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();

  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = tid; i < size; i += blockDim.x) {
    output[i] = scale * input[i] * weight[i];
  }
}

static __global__ void row_rmsnorm_f32_veterize(
    float* in,
    float* wei,
    float* out,
    int size,
    float eps) {
  const int tid = threadIdx.x;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);

  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }

  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(
        scale * in_float4.x * wei_float4.x,
        scale * in_float4.y * wei_float4.y,
        scale * in_float4.z * wei_float4.z,
        scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

void rmsnorm_kernel_cuda(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(
      input.device_type() == base::DeviceType::kDeviceCUDA &&
      weight.device_type() == base::DeviceType::kDeviceCUDA &&
      output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-5f;
  const int32_t size = static_cast<int32_t>(input.size());
  float* input_ptr = const_cast<float*>(input.ptr<float>());
  float* weight_ptr = const_cast<float*>(weight.ptr<float>());
  float* output_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_veterize<<<1, threads_num, 0, stream_>>>(
        input_ptr, weight_ptr, output_ptr, size, eps);
  } else {
    row_rmsnorm_f32_veterize<<<1, threads_num>>>(
        input_ptr, weight_ptr, output_ptr, size, eps);
  }
}
} // namespace kernel