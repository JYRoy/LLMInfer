#include "base/base.h"
#include <string>

namespace base {
Status::Status(int code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}
namespace error {
Status PathNotValid(const std::string& err_msg) {
  return Status{kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
  return Status{kModelParseError, err_msg};
}
} // namespace error
} // namespace base