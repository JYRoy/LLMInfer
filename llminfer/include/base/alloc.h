#ifndef __BASE_ALLOC_H__
#define __BASE_ALLOC_H__

#include <iostream>
#include <map>
#include <memory>
#include "base.h"

namespace base {
enum class MemcpyKind {
  kMemcpyH2H = 0,
  kMemcpyH2D = 1,
  kMemcpyD2H = 2,
  kMemcpyD2D = 3,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type)
      : device_type_(device_type) {}

  virtual DeviceType device_type() const {
    return device_type_;
  }

  virtual void release(void* ptr) const = 0;

  virtual void* allocate(size_t byte_size) const = 0;

  virtual void memcpy(
      const void* src_ptr,
      void* dest_ptr,
      size_t byte_size,
      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyH2H,
      void* stream = nullptr,
      bool need_sync = false) const;

  //   virtual void memset_zero(
  //       void* ptr,
  //       size_t byte_size,
  //       void* stream,
  //       bool need_sync = false);

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};

class DeviceAlloctorFactory {
 public:
  static std::shared_ptr<DeviceAllocator> get_instance(
      base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
      return CPUDeviceAllocatorFactory::get_instance();
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
      return CPUDeviceAllocatorFactory::get_instance();
    } else {
      LOG(FATAL) << "This device type of allocator is not supported!";
      return nullptr;
    }
  }
};

} // namespace base

#endif