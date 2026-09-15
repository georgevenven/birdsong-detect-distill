#!/usr/bin/env bash
set -euo pipefail
helper_root=/media/george-vengrovski/disk2/birdsong-powdermill-worker-20260911
helper_cmake=/home/george-vengrovski/llama.cpp/.cmake-venv/bin/cmake
"$helper_cmake" -S "$helper_root/llama.cpp" -B "$helper_root/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
"$helper_cmake" --build "$helper_root/build" --target llama-server -j 4
"$helper_root/build/bin/llama-server" --version
