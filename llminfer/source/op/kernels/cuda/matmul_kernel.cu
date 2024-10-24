#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"

namespace kernel {
// sgemv
// 权重是 N * N 的矩阵
// 输入是 N * 1 的张量
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int K) {
  // 用于存储每个线程的中间数据
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  // 因为设置的 block 数目是行数
  // 所以每一个 block 负责处理一个行
  // start_row 理论上等于 end_row
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  // 按照 float4 进行向量化处理
  constexpr int pack_size = 4;
  // 按照 float4 处理之后还剩下 pack_num 组数据
  const int pack_num = M / pack_size;
  // 因为 M 不一定能被 pack_size 整除
  // 所以 pack_size * pack_num 只能处理 pack_off 个具体的数据
  const int pack_off = pack_size * pack_num;

#pragma unroll
  // 按照行进行处理
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    // 向量化处理转为 float4 的类型
    float4* input_float4_ptr = (float4*)input;
    // 因为 weight 是矩阵，所以要定位到一行和 input 的一行相乘
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    // 四个一组的处理数据，为了访存合并
    // 线程需要按照步长是 blockDim.x（即线程数）来访问数据
    // 这样个 warp 的线程，同时访问的数据在内存上是连续的
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      // 四个数据进行乘加的 reduce
      float part_sum = input_float4.x * weight_float4.x +
          input_float4.y * weight_float4.y + input_float4.z * weight_float4.z +
          input_float4.w * weight_float4.w;
      // 计算完成后存储到当前线程的临时存储中
      sdata[tid] += part_sum;
    }
    // 对应剩余的数据，逐一进行处理
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    // 对于已经进行完按位乘的线程，已经存储到了长度为 128 的 sdata 数组中
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    // 进行完 block reduce，第一个线程的持有的数据就是输出中的一个位置的元素
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      // 将最终结果存储到 output 数组上
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    const float scale,
    const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0); // row
  const int32_t M = weight.get_dim(1); // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);

  CHECK_EQ(M, input.get_dim(0));
  // K 个 block，每个 block 负责处理一行
  // 128 个线程，对应 4 个 warp，每个 warp 负责处理一行中的一部分数据
  if (config && config->stream) {
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(),
        weight.ptr<float>(),
        const_cast<float*>(output.ptr<float>()),
        M,
        K);
  } else {
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(
        input.ptr<float>(),
        weight.ptr<float>(),
        const_cast<float*>(output.ptr<float>()),
        M,
        K);
  }
}

} // namespace kernel