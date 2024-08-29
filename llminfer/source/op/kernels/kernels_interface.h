#ifndef __KERNELS_INTERFACE_H__
#define __KERNELS_INTERFACE_H__
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
typedef void (*RMSNormKernel)(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    void* stream);
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
} // namespace kernel

#endif