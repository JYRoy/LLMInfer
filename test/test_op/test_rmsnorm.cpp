#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

TEST(test_rmsnorm_cu, rmsnorm_with_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32;

  tensor::Tensor input_cpu(
      base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor weight_cpu(
      base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor output_cpu(
      base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    input_cpu.index<float>(i) = dist(mt);
    weight_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor input_cuda = input_cpu.clone();
  tensor::Tensor weight_cuda = weight_cpu.clone();
  tensor::Tensor output_cuda = output_cpu.clone();
  input_cuda.to_cuda(nullptr);
  weight_cuda.to_cuda(nullptr);
  output_cuda.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(
      input_cuda, weight_cuda, output_cuda, stream);
  output_cuda.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(
      input_cpu, weight_cpu, output_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output_cuda.index<float>(i), output_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}