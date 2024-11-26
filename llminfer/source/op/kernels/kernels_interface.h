#ifndef __KERNELS_INTERFACE_H__
#define __KERNELS_INTERFACE_H__
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
typedef void (*AddKernel)(
    const tensor::Tensor& input1,
    const tensor::Tensor& input2,
    const tensor::Tensor& output,
    void* stream);
AddKernel get_add_kernel(base::DeviceType device_type);
typedef void (*RMSNormKernel)(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    void* stream);
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

typedef void (*MatmulKernel)(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& output,
    float scale,
    const CudaConfig* config);
MatmulKernel get_matmul_kernel(base::DeviceType device_type);

typedef void (*SwigluKernel)(
    const tensor::Tensor& input1,
    const tensor::Tensor& input2,
    const tensor::Tensor& output,
    void* stream);
SwigluKernel get_swiglu_kernel(
    base::DeviceType device_type,
    void* stream = nullptr);

typedef void (*MHAKernel)(
    int32_t pos,
    int32_t head_num,
    int32_t layer_index,
    int32_t seq_len,
    int32_t kv_dim,
    int32_t kv_mul,
    int32_t head_size,
    const tensor::Tensor& mha_out,
    const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor,
    const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor,
    base::DeviceType device_type,
    CudaConfig*);
MHAKernel get_mha_kernel(base::DeviceType device_type);
} // namespace kernel

#endif