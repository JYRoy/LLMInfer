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
