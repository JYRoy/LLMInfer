#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op {
BaseLayer::BaseLayer(
    base::DeviceType device_type,
    LayerType layer_type,
    base::DataType data_type,
    std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const {
  return data_type_;
}

LayerType BaseLayer::layer_type() const {
  return layer_type_;
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) {
  inputs_.resize(size);
}

void Layer::reset_output_size(size_t size) {
  outputs_.resize(size);
}

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const {
  return cuda_config_;
}

size_t Layer::input_size() const {
  return inputs_.size();
}

size_t Layer::output_size() const {
  return outputs_.size();
}

base::Status Layer::forward(
    const tensor::Tensor& input1,
    const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(
    const tensor::Tensor& input1,
    const tensor::Tensor& input2,
    const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(
    const tensor::Tensor& input1,
    const tensor::Tensor& input2,
    const tensor::Tensor& input3,
    const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(
    const tensor::Tensor& input1,
    const tensor::Tensor& input2,
    const tensor::Tensor& input3,
    const tensor::Tensor& input4,
    const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(
    const tensor::Tensor& input1,
    const tensor::Tensor& input2,
    const tensor::Tensor& input3,
    const tensor::Tensor& input4,
    const tensor::Tensor& input5,
    const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::check() const {
  return base::error::FunctionNotImplement(
      "The check function is not implement yet");
}

Layer::Layer(
    base::DeviceType device_type,
    LayerType layer_type,
    std::string layer_name)
    : BaseLayer(
          device_type,
          layer_type,
          base::DataType::kDataTypeFp32,
          std::move(layer_name)) {}

base::Status Layer::init() {
  return base::error::Success();
}

base::Status Layer::forward() {
  return base::error::FunctionNotImplement("");
}

base::Status Layer::check_tensor(
    const tensor::Tensor& tensor,
    base::DeviceType device_type,
    base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(
    const tensor::Tensor& tensor,
    base::DeviceType device_type,
    base::DataType data_type,
    ...) const {
  // 定义一个va_list类型的变量，变量是指向参数的指针
  std::va_list args;
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  // va_start初始化刚定义的变量，第二个参数是最后一个显式声明的参数。
  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  for (int32_t i = 0; i < dims; ++i) {
    // va_arg返回变长参数的值，第二个参数是该变长参数的类型。
    // 从参数列表中逐个取出数据, 取出数据的类型由 type 决定, 它返回这个 type
    // 类型的值, 你可以马上把它赋值给另一个变量.
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.get_dim(i)) {
      return base::error::InvalidArgument(
          "The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  // va_end将va_start定义的变量重置为NULL。
  va_end(args);
  return base::error::Success();
}

} // namespace op