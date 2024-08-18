#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>

namespace tensor {
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

Tensor::Tensor(
    base::DataType data_type,
    int32_t dim0,
    bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc,
    void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is is true when ptr parameter is not a null pointer.";
      init_buffer(alloc, data_type_, need_alloc, ptr);
    }
  }
}

Tensor::Tensor(
    base::DataType data_type,
    int32_t dim0,
    int32_t dim1,
    bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc,
    void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(
    base::DataType data_type,
    int32_t dim0,
    int32_t dim1,
    int32_t dim2,
    bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc,
    void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(
    base::DataType data_type,
    int32_t dim0,
    int32_t dim1,
    int32_t dim2,
    int32_t dim3,
    bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc,
    void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(
    base::DataType data_type,
    std::vector<int32_t> dims,
    bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc,
    void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}
void Tensor::init_buffer(
    std::shared_ptr<base::DeviceAllocator> alloc,
    base::DataType data_type,
    bool need_alloc,
    void* ptr) {
  if (!alloc && !need_alloc) {
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
        DataTypeSize(data_type) * size_, nullptr, ptr, true);
    this->buffer_ = buffer;
  } else {
    allocate(alloc, true);
  }
}

bool Tensor::allocate(
    std::shared_ptr<base::DeviceAllocator> allocator,
    bool need_realloc) {
  if (!allocator) {
    LOG(ERROR)
        << "The allocator parameter in the allocate function is null pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR)
        << "The byte_size parameter in the allocate function is equal to zero!";
    return false;
  }

  if (buffer_ && byte_size <= buffer_->byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }

  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  if (!buffer_->ptr()) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

size_t Tensor::size() const {
  return this->size_;
}

size_t Tensor::byte_size() const {
  return this->size() * DataTypeSize(data_type_);
}

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

base::DeviceType Tensor::device_type() const {
  if (!buffer_) {
    return base::DeviceType::kDeviceUnknown;
  }
  return buffer_->device_type();
}

bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

} // namespace tensor