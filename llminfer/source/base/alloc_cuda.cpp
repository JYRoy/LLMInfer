#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {

CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }

  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, byte_size);
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  if (ptr) {
    cudaFree(ptr);
  }
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance =
    nullptr;

} // namespace base
