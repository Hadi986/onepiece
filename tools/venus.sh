#!/bin/bash


CHIP_LIB_PATH="/data_bak/nfs/cywang/yolov8/lib"
# 更新脚本开始

# 切换到指定目录
cd /data_bak/sfyan/ivse/prebuilt_nna2

# 拉取最新代码
git pull

# 复制 AIP_T41 相关文件
cp Magik/InferenceKit/third_party/AIP_T41/5.1.8/7.2.0/2.29/lib/uclibc/* Magik/InferenceKit/venus/mips/T41/public/7.2.0/2.29/lib/uclibc/

# 复制 DRIVERS 相关文件
cp Magik/InferenceKit/third_party/DRIVERS/T41/5.1.9/7.2.0/2.29/lib/uclibc/* Magik/InferenceKit/venus/mips/T41/public/7.2.0/2.29/lib/uclibc/

# 复制到芯片库路径
cp Magik/InferenceKit/venus/mips/T41/public/7.2.0/2.29/lib/uclibc/* $CHIP_LIB_PATH

# 更新脚本结束
echo "cp lib over"