#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

namespace kernel {

static __global__ void row_rmsnorm_f32(
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
    row_rmsnorm_f32<<<1, threads_num, 0, stream_>>>(
        input_ptr, weight_ptr, output_ptr, size, eps);
  } else {
    row_rmsnorm_f32<<<1, threads_num>>>(
        input_ptr, weight_ptr, output_ptr, size, eps);
  }
}
} // namespace kernel