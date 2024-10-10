#include "base/base.h"
#include <string>

namespace base {
Status::Status(int code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}

namespace error {
Status Success(const std::string& err_msg) {
  return Status{kSuccess, err_msg};
}

Status FunctionNotImplement(const std::string& err_msg) {
  return Status{kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
  return Status{kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
  return Status{kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
  return Status{kInternalError, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
  return Status{kInvalidArgument, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
  return Status{kKeyValueHasExist, err_msg};
}
} // namespace error
} // namespace base