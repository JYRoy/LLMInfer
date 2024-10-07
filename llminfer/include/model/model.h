#ifndef _INCLUDE_MODEL_MODEL_H_
#define _INCLUDE_MODEL_MODEL_H_
#include <map>
#include <string>
#include "base/base.h"
#include "config.h"

namespace model {
class Model {
 protected:
  virtual base::Status read_model_file();

 protected:
  bool is_quant_model_ = false;

  std::string token_path_;
  std::string model_path_;
};
} // namespace model

#endif