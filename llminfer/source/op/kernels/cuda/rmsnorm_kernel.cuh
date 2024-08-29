#ifndef __RMSNORM_KERNEL_CU_H__
#define __RMSNORM_KERNEL_CU_H__
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cuda(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    void* stream = nullptr);
} // namespace kernel
#endif // __RMSNORM_KERNEL_H__
