@echo off

@REM REM 设置代理（如果需要）
@REM set PROXY=http://127.0.0.1:7890

@REM set HTTPS_PROXY=%PROXY%
@REM set HTTP_PROXY=%PROXY%
@REM set ALL_PROXY=%PROXY%

REM Release libtorch2.4.0+cpu
@REM curl -L -o release.zip "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%%2Bcpu.zip"
REM Debug libtorch2.4.0+cpu
@REM curl -L -o debug.zip "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.4.0%%2Bcpu.zip"
REM Release libtorch2.4.0+cu121
@REM curl -L -o release.zip "https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.4.0%%2Bcu121.zip"
REM Debug libtorch2.4.0+cu121
@REM curl -L -o debug.zip "https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-debug-2.4.0%%2Bcu121.zip"

if exist release.zip (
    echo release installing...
    mkdir windows\Release
    tar -xf release.zip
    xcopy libtorch\* windows\Release\ /E /I /H /Y
    del release.zip
    echo release installed
)

if exist debug.zip (
    echo debug installing...
    mkdir windows\Debug
    tar -xf debug.zip
    xcopy libtorch\* windows\Debug\ /E /I /H /Y
    @REM del debug.zip
    echo debug installed
)


echo finish!
pause

