#!/usr/bin/env bash
set -euo pipefail
worker_root=/media/george/DATA/birdsong-powdermill-worker-20260913
cd "$worker_root"
sha256sum -c source.sha256
mkdir -p llama.cpp
tar -xzf llama-source.tar.gz -C llama.cpp
cmake -S llama.cpp -B build-sm70 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build build-sm70 --target llama-server -j6
build-sm70/bin/llama-server --version
