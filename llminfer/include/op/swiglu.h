#ifndef _INCLUDE_OP_SWIGLU_H_
#define _INCLUDE_OP_SWIGLU_H_
#include "layer.h"
namespace op {
class SwiGLULayer : public op::Layer {
 public:
  explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t hidden_dim_ = 0;
};
} // namespace op
#endif // _INCLUDE_OP_SWIGLU_H_
