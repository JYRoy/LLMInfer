# LLMInfer

自制的大模型推理框架

## 目录

- [LLMInfer](#llminfer)
  - [目录](#目录)
  - [环境要求](#环境要求)
  - [安装方式](#安装方式)

## 环境要求

- C++ 17+
- GCC 11.4.0
- armadill0 14.0.2
- googletest 1.15.2
- glog 0.7.X (commit id b3b9eb9)
- setencepiece 0.2.X (commit id 499380)

## 安装方式

```shell
mkdir build
cd build
cmake ..
make -j10
```