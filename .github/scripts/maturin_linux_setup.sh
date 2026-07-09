#!/usr/bin/env bash

echo Installing LLVM via yum
yum update -y
yum -y install \
    llvm-devel-21.1.8 \
    clang-devel-21.1.8 \
    libzstd-devel \
    zstd \
    ncurses-devel \
    zlib-devel \
    libffi \
    libffi-devel \
    libxml2-devel

export LLVM_DIR=/usr
export LLVM_SYS_211_PREFIX=/usr

echo Installing SparseSuite via yum
yum install -y suitesparse-devel
