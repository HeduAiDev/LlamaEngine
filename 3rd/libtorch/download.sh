#!/bin/bash

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
