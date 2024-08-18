#ifndef __BASE_BASE_H_
#define __BASE_BASE_H_

#include <cstdint>
#include <string>

namespace base {
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};
} // namespace base

#endif