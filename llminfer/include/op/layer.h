#ifndef _INCLUDE_OP_LAYER_H_
#define _INCLUDE_OP_LAYER_H_
#include <base/cuda_config.h>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op {
class Layer;
enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

class BaseLayer {
 public:
  explicit BaseLayer(
      base::DeviceType device_type,
      LayerType layer_type,
      base::DataType data_type,
      std::string layer_name = "");

  base::DataType data_type() const;

  LayerType layer_type() const;

  virtual base::Status init() = 0;

  virtual base::Status forward() = 0;

  virtual base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& output1) = 0;

  virtual base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& output1) = 0;

  virtual base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& input3,
      const tensor::Tensor& output1) = 0;

  virtual base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& input3,
      const tensor::Tensor& input4,
      const tensor::Tensor& output1) = 0;

  virtual base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& input3,
      const tensor::Tensor& input4,
      const tensor::Tensor& input5,
      const tensor::Tensor& output1) = 0;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t input_size() const = 0;

  virtual size_t output_size() const = 0;

  virtual base::Status check() const = 0;

  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

  const std::string& get_layer_name() const;

  void set_layer_name(const std::string& layer_name);

  void set_device_type(base::DeviceType device_type);

  base::DeviceType device_type() const;

 protected:
  std::string layer_name_; // 层名
  LayerType layer_type_ = LayerType::kLayerUnknown; // 层类型
  base::DataType data_type_ = base::DataType::kDataTypeUnknown; // 层数据类型
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown; // 设备类型
};

class Layer : public BaseLayer {
 public:
  explicit Layer(
      base::DeviceType device_type,
      LayerType layer_type,
      std::string layer_name = "");

  base::Status init() override;

  base::Status check_tensor(
      const tensor::Tensor& tensor,
      base::DeviceType device_type,
      base::DataType data_type) const;

  base::Status check_tensor_with_dim(
      const tensor::Tensor& tensor,
      base::DeviceType device_type,
      base::DataType data_type,
      ...) const;

  base::Status check() const override;

  base::Status forward() override;

  base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& output1) override;

  base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& output1) override;

  base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& input3,
      const tensor::Tensor& output1) override;

  base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& input3,
      const tensor::Tensor& input4,
      const tensor::Tensor& output1) override;

  base::Status forward(
      const tensor::Tensor& input1,
      const tensor::Tensor& input2,
      const tensor::Tensor& input3,
      const tensor::Tensor& input4,
      const tensor::Tensor& input5,
      const tensor::Tensor& output1) override;

  // idx 表示第几（idx）个输入
  void set_input(int32_t idx, const tensor::Tensor& input) override; // 传入输入

  void set_output(int32_t idx, const tensor::Tensor& output)
      override; // 传入输出

  const tensor::Tensor& get_input(int32_t idx) const override; // 获取输入

  const tensor::Tensor& get_output(int32_t idx) const override; // 获取输出

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  size_t input_size() const override; // 获取输入的个数

  size_t output_size() const override; // 获取输出的个数

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

  virtual void to_cuda();

  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

  std::shared_ptr<kernel::CudaConfig> cuda_config() const;

 protected:
  std::vector<tensor::Tensor> inputs_; // 存放输入的数组
  std::vector<tensor::Tensor> outputs_; // 存放输出的数组
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
};
} // namespace op

#endif