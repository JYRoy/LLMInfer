#ifndef _INCLUDE_MODEL_MODEL_H_
#define _INCLUDE_MODEL_MODEL_H_
#include <map>
#include <string>
#include "config.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "tensor/tensor.h"

namespace model {
class Model {
 public:
  explicit Model(
      base::ModelType model_type,
      std::string token_path,
      std::string model_path,
      bool is_quant_model);

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

 protected:
  virtual base::Status read_model_file();

  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(
      int32_t layer_idx,
      int32_t token_pos) const = 0;

  virtual void create_param_layers() = 0;

 protected:
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
};
} // namespace model

#endif