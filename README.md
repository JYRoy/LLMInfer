# LLMInfer
An Inference Acceleration Framework for Large Language Model

## Contents

- [LLMInfer](#llminfer)
  - [Contents](#contents)
  - [Requirments](#requirments)
  - [Installation](#installation)

## Requirments

- C++ 17+
- GCC 11.4.0
- armadill0 14.0.2
- googletest 1.15.2
- glog 0.7.X (commit id b3b9eb9)
- setencepiece 0.2.X (commit id 499380)

## Installation

```shell
mkdir build
cd build
cmake ..
make -j10
```