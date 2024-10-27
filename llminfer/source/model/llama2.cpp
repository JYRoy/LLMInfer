#include "model/llama2.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/rmsnorm.h>
#include <utility>

namespace model {
LLama2Model::LLama2Model(
    std::string token_path,
    std::string model_path,
    bool is_quant_model)
    : Model(
          base::ModelType::kModelTypeLLama2,
          std::move(token_path),
          std::move(model_path),
          is_quant_model) {}

std::pair<tensor::Tensor, tensor::Tensor> LLama2Model::slice_kv_cache(
    int32_t layer_idx,
    int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr = const_cast<float*>(
      get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr = const_cast<float*>(
      get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  auto key_cache = std::make_shared<base::Buffer>(
      config_->kv_dim_ * sizeof(float), nullptr, key_cache_ptr, true);
  auto val_cache = std::make_shared<base::Buffer>(
      config_->kv_dim_ * sizeof(float), nullptr, val_cache_ptr, true);
  key_cache->set_device_type(device_type_);
  val_cache->set_device_type(device_type_);
  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_);
  key.assign(key_cache);
  val.assign(val_cache);
  return {key, val};
}

// void LLama2Model::create_param_layers() {
//   CHECK(!is_quant_model_);
//   CHECK(llama_layers_ != nullptr);
//   // The embedding layer
//   auto cpu_device_type = base::DeviceType::kDeviceCPU;
//   llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
//       device_type_,
//       config_->dim_,
//       config_->seq_len_,
//       std::abs(config_->vocab_size_));

//   const void* weight_embedding = raw_model_data_->weight(0);
//   llama_layers_->embedding_layer_->set_weight(
//       0,
//       {std::abs(config_->vocab_size_), config_->dim_},
//       weight_embedding,
//       cpu_device_type);

//   // create all matmul layer (linear op)
//   int32_t dim = config_->dim_;
//   size_t pos = dim * std::abs(config_->vocab_size_) + dim *
//   config_->layer_num_;

//   // create weight matrix for query
//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
//     wq->set_weight(
//         0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
//     llama_layers_->wq_layers_.push_back(wq);
//     pos += dim * dim;
//   }

//   // create weight matrix for key
//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     auto wk =
//         std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_,
//         dim);
//     wk->set_weight(
//         0,
//         {config_->kv_dim_, dim},
//         this->raw_model_data_->weight(pos),
//         cpu_device_type);
//     llama_layers_->wk_layers_.push_back(wk);
//     pos += config_->kv_dim_ * dim;
//   }

//   // create weight matrix for value
//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     auto wv =
//         std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_,
//         dim);
//     wv->set_weight(
//         0,
//         {config_->kv_dim_, dim},
//         this->raw_model_data_->weight(pos),
//         cpu_device_type);
//     llama_layers_->wv_layers_.push_back(wv);
//     pos += config_->kv_dim_ * dim;
//   }

//   // create weight matrix for output
//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
//     wo->set_weight(
//         0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
//     llama_layers_->wo_layers_.push_back(wo);
//     pos += dim * dim;
//   }

//   // skip ffn rmsnorm
//   pos += config_->layer_num_ * dim;

//   // w1 layers
//   int32_t hidden_dim = config_->hidden_dim_;
//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim,
//     dim); w1->set_weight(
//         0,
//         {hidden_dim, dim},
//         this->raw_model_data_->weight(pos),
//         cpu_device_type);
//     llama_layers_->w1_layers_.push_back(w1);
//     pos += dim * hidden_dim;
//   }

//   // w2 layers
//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim,
//     hidden_dim); w2->set_weight(
//         0,
//         {dim, hidden_dim},
//         this->raw_model_data_->weight(pos),
//         cpu_device_type);
//     llama_layers_->w2_layers_.push_back(w2);
//     pos += dim * hidden_dim;
//   }

//   // w3 layers
//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim,
//     dim); w3->set_weight(
//         0,
//         {hidden_dim, dim},
//         this->raw_model_data_->weight(pos),
//         cpu_device_type);
//     llama_layers_->w3_layers_.push_back(w3);
//     pos += dim * hidden_dim;
//   }

//   // skip final rms weight
//   pos += dim;
//   pos += config_->seq_len_ * config_->head_size_;

//   llama_layers_->cls_layer_ = std::make_shared<op::MatmulLayer>(
//       device_type_, config_->vocab_size_, dim);

//   if (config_->is_shared_weight_) {
//     // using token embedding weight
//     llama_layers_->cls_layer_->set_weight(
//         0,
//         {config_->vocab_size_, dim},
//         this->raw_model_data_->weight(0),
//         cpu_device_type);
//   } else {
//     llama_layers_->cls_layer_->set_weight(
//         0,
//         {config_->vocab_size_, dim},
//         this->raw_model_data_->weight(pos),
//         cpu_device_type);
//   }

//   // create rmsnorm layer
//   size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
//         std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

//     const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
//     rms_norm_layer->set_weight(
//         0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
//     llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
//     rmsnorm_pos += config_->dim_;
//   }

//   rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
//   rmsnorm_pos += config_->layer_num_ * config_->dim_ *
//       (config_->kv_head_num_ * config_->head_size_);
//   rmsnorm_pos += config_->layer_num_ * config_->dim_ *
//       (config_->kv_head_num_ * config_->head_size_);
//   rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

//   for (int32_t i = 0; i < config_->layer_num_; ++i) {
//     std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
//         std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
//     const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
//     rms_norm_layer->set_weight(
//         0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
//     llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

//     rmsnorm_pos += config_->dim_;
//   }

//   rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
//   rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
//   rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

//   std::shared_ptr<op::RmsNormLayer> rms_final_layer =
//       std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

//   const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
//   rms_final_layer->set_weight(
//       0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
//   llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
// }

// void LLama2Model::create_nonparam_layers() {
//   // 不带权重的层
//   // 直接创建不带权重的算子层，不需要读取权重
//   CHECK(llama_layers_ != nullptr);
//   llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
//       device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

//   llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
//       device_type_,
//       0,
//       config_->kv_mul_,
//       config_->kv_dim_,
//       config_->seq_len_,
//       config_->head_num_,
//       config_->head_size_);

//   llama_layers_->add_layer_ =
//   std::make_shared<op::VecAddLayer>(device_type_);

//   llama_layers_->swiglu_layer_ =
//       std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
// }

} // namespace model