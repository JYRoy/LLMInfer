#include "rmsnorm_kernel.h"

namespace kernel {
void rmsnorm_kernel_cpu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(
      input.device_type() == base::DeviceType::kDeviceCPU &&
      weight.device_type() == base::DeviceType::kDeviceCPU &&
      output.device_type() == base::DeviceType::kDeviceCPU);

  const float* input_ptr = input.ptr<float>();
  float* output_ptr = const_cast<float*>(output.ptr<float>());

  int size = static_cast<int32_t>(input.size());
  float square_sum = 0.f;
  for (int i = 0; i < size; i++) {
    float input_value = input.index<float>(i);
    square_sum += input_value * input_value;
  }

  const float eps = 1e-5f;
  float scale = square_sum / float(size);
  float rsqrt = 1.f / std::sqrt(scale + eps);
  for (int i = 0; i < size; i++) {
    *(output_ptr + i) = weight.index<float>(i) * (rsqrt * (*(input_ptr + i)));
  }
}
} // namespace kernel