#include "model/llama2.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
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

void LLama2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_,
      config_->dim_,
      config_->seq_len_,
      std::abs(config_->vocab_size_));

  const void* weight_embedding = raw_model_data_->weight(0);
  llama_layers_->embedding_layer_->set_weight(
      0,
      {std::abs(config_->vocab_size_), config_->dim_},
      weight_embedding,
      cpu_device_type);

  // create all matmul layer (linear op)
  int32_t dim = config_->dim_;
  // 逐层读取 query layer，开始的偏移是 dim × vocab size + N × dim
  // 从 export.py 中可知，wq 的 前面有 token embedding 和 attention norm
  // 的权重，因此要将 pos 偏移到对应的位置
  // N 是 transformer 块的个数，当输出权重模型文件后权重有如下的排布：
  // ---------------
  // token embedding   1 × dim × vocab size
  // ---------------
  // attention norm    N × dim
  // ---------------
  // weight query      N × dim × dim <==== pos
  // ---------------
  // weight key        N × dim × dim
  // ---------------
  // weight value      N × dim × dim
  // ---------------
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;

  // create weight matrix for query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    // 创建一个新的 matmul 层
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    // 每个 wq layer 的维度是 dim×dim
    wq->set_weight(
        0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    // 添加每个层的 wq 到 llama 的 layer 数组中
    llama_layers_->wq_layers_.push_back(wq);
    // pos指向下一个wq layer权重的开始
    pos += dim * dim;
  }

  // create weight matrix for key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk =
        std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(
        0,
        {config_->kv_dim_, dim},
        this->raw_model_data_->weight(pos),
        cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv =
        std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(
        0,
        {config_->kv_dim_, dim},
        this->raw_model_data_->weight(pos),
        cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(
        0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(
        0,
        {hidden_dim, dim},
        this->raw_model_data_->weight(pos),
        cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(
        0,
        {dim, hidden_dim},
        this->raw_model_data_->weight(pos),
        cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(
        0,
        {hidden_dim, dim},
        this->raw_model_data_->weight(pos),
        cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  pos += config_->seq_len_ * config_->head_size_;

  llama_layers_->cls_layer_ = std::make_shared<op::MatmulLayer>(
      device_type_, config_->vocab_size_, dim);

  if (config_->is_shared_weight_) {
    // using token embedding weight
    llama_layers_->cls_layer_->set_weight(
        0,
        {config_->vocab_size_, dim},
        this->raw_model_data_->weight(0),
        cpu_device_type);
  } else {
    llama_layers_->cls_layer_->set_weight(
        0,
        {config_->vocab_size_, dim},
        this->raw_model_data_->weight(pos),
        cpu_device_type);
  }

  // create rmsnorm layer
  // first rmsnorm layer
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(
        0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->dim_ *
      (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ *
      (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(
        0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  // final rmsnorm
  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(
      0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

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

void LLama2Model::attention_qkv(
    int32_t layer_idx,
    const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  // rmsnorm_output @ wq = query
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr)
      << "The query layer in the attention block is null pointer.";
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  // 将输入的 embedding 向量和 query layer 自身的权重相乘并将计算结果存储在
  // query 变量中
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  // rmsnorm_output @ wk = key，就是当前的最后一列K矩阵
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr)
      << "The key layer in the attention block is null pointer.";
  // 将输入的 embedding 向量和 key layer 自身的权重相乘并将计算结果存储在 key
  // 变量中
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // value
  // rmsnorm_output @ wv = value，就是当前的最后一行的V矩阵
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr)
      << "The value layer in the attention block is null pointer.";
  // 将输入的 embedding 向量和 value layer 自身的权重相乘并将计算结果存储在
  // value 变量中
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query,
      key,
      pos_tensor,
      get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache),
      tensor::Tensor{}));
}

} // namespace model