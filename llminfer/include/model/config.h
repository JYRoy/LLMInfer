#ifndef _INCLUDE_MODEL_LLAMA_CONFIG_H__
#define _INCLUDE_MODEL_LLAMA_CONFIG_H_
namespace model {
struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
};

} // namespace model
#endif // _INCLUDE_MODEL_LLAMA_CONFIG_H_
