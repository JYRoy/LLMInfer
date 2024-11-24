#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
namespace model {
Model::Model(
    base::ModelType model_type,
    std::string token_path,
    std::string model_path,
    bool is_quant_model)
    : model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid(
        "Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid(
        "Failed to open the weight file " + model_path_ +
        " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid(
        "Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }
  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }
  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = ftell(file);
  fclose(file);

  raw_model_data_->fd = fd;
  raw_model_data_->data = mmap(
      nullptr,
      raw_model_data_->file_size,
      PROT_READ,
      MAP_PRIVATE,
      raw_model_data_->fd,
      0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError(
        "Failed to map the weight file " + model_path_ + " into memory.");
  }
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) +
        sizeof(ModelConfig) + sizeof(group_size_);
  }
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError(
        "Failed to map the weight file " + model_path_ +
        " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;

  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  if (std::abs(config.vocab_size) != config_->vocab_size_) {
    return base::error::ModelParseError(
        "Vocabulary size mismatch between the model file and the token list.");
  }
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

} // namespace model
