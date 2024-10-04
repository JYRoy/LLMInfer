#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "tensor/tensor.h"

TEST(test_tensor, init1) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc);
  ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, init2) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, false, alloc);
  ASSERT_EQ(t1.is_empty(), true);
}

TEST(test_tensor, init3) {
  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, nullptr);
  ASSERT_EQ(t1.is_empty(), true);
}

TEST(test_tensor, init4) {
  float* ptr = new float[32];
  ptr[0] = 31;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.ptr<float>(), ptr);
  ASSERT_EQ(*t1.ptr<float>(), 31);
}

TEST(test_tensor, index) {
  float* ptr = new float[32];
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  ptr[0] = 32;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
  void* p1 = t1.ptr<void>();
  p1 += 1;
  float* f1 = t1.ptr<float>();
  f1 += 1;
  ASSERT_NE(f1, p1);
  delete[] ptr;
}

TEST(test_tensor, dims_stride) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, 32, 3, true, alloc);

  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.get_dim(0), 32);
  ASSERT_EQ(t1.get_dim(1), 32);
  ASSERT_EQ(t1.get_dim(2), 3);

  const auto& strides = t1.strides();
  ASSERT_EQ(strides.at(0), 32 * 3);
  ASSERT_EQ(strides.at(1), 3);
  ASSERT_EQ(strides.at(2), 1);
}

TEST(test_tensor, to_cu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  float* p1 = t1_cpu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    *(p1 + i) = 1.f;
  }

  t1_cpu.to_cuda();
  float* p2 = new float[32 * 32];
  cudaMemcpy(
      p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(p2 + i), 1.f);
  }
  delete[] p2;
}