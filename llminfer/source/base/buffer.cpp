#include "base/buffer.h"

namespace base {
Buffer::Buffer(
    size_t byte_size,
    std::shared_ptr<DeviceAllocator> allocator,
    void* ptr,
    bool use_external) {
  if (!ptr_ && allocator_) {
    device_type_ = allocator_->device_type();
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size);
  }
}

Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);
      ptr_ = nullptr;
    }
  }
}

bool Buffer::allocate() {
  if (allocator_ && byte_size_ != 0) {
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size_);
    if (!ptr_) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

void* Buffer::ptr() {
  return ptr_;
}

const void* Buffer::ptr() const {
  return ptr_;
}

size_t Buffer::byte_size() const {
  return byte_size_;
}

bool Buffer::is_external() const {
  return this->use_external_;
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
  return allocator_;
}

} // namespace base
