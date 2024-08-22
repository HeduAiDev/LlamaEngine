#!/bin/bash
# linux端
# 如果已经安装了 pytorch 可以直接使用下面的指令，无需额外安装libtorch
# cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -B build

# 下载libtorch
# 设置代理（如果需要）
# PROXY="http://127.0.0.1:7890"

# 设置环境变量以使用代理（如果需要）
# export HTTPS_PROXY=$PROXY
# export HTTP_PROXY=$PROXY
# export ALL_PROXY=$PROXY

# libtorch 2.4.0 + cpu
# curl -L -o release.zip "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.4.0%2Bcpu.zip"
# libtorch 2.4.0 + cu121
curl -L -o release.zip "https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.4.0%2Bcu121.zip"

if [ -f "release.zip" ]; then
    echo "release installing..."
    mkdir -p linux/Release
    unzip -q release.zip -d .
    mv libtorch/* linux/Release/
    rm -r libtorch
    rm release.zip
    echo "release installed"
fi

if [ -f "debug.zip" ]; then
    echo "debug installing..."
    mkdir -p linux/Debug
    unzip -q debug.zip -d .
    mv libtorch/* linux/Debug/
    rm -r libtorch
    rm debug.zip
    echo "debug installed"
fi

echo "finish!"
