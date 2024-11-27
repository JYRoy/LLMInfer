#include "emb_kernel.cuh"
namespace kernel {
__global__ void emb_kernel_cu_fp32(
    int32_t vocab_size,
    int32_t token_num, // token 个数
    int32_t weight_dim, // embedding 的大小
    const int32_t* input_ptr, // token 的 id
    const float* weight_ptr, // embedding 词表
    float* output_ptr) {
  // 获取当前 block 负责的单词 id
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }

  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  // 获取当前 embedding 的起始位置
  const float* weight_ptr_start = weight_ptr + token * weight_dim;
  // 定位到词表的一行（即一个 token 的 embedding 之后），开始搬运
  // 每个线程只负责一部分数据
  // blockDim.x 为 128，则这 128 个线程的访存是连续的
  // 每个线程要去访问自己负责的下一个元素时，要 + 128
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    int32_t vocab_size,
    void* stream) {
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  // 每个 block 负责一个单词
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, stream_>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  } else {
    emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  }
}
} // namespace kernel