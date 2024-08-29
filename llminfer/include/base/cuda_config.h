#ifndef __CUDA_CONFIG_H__
#define __CUDA_CONFIG_H__
#include <cuda_runtime_api.h>

namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
} // namespace kernel
#endif // __CUDA_CONFIG_H__
