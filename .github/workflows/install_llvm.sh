#!/usr/bin/env bash
if [ -f "/etc/debian_version" ]; then
    apt update
    apt install -y wget lsb-release software-properties-common gnupg
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    ./llvm.sh 17
    apt install -y libpolly-17-dev libzstd-dev zstd libncurses-dev zlib1g-dev
    export LLVM_DIR=/usr/lib/llvm-17
    export LLVM_SYS_170_PREFIX=/usr/lib/llvm-17
else
    # must be AlmaLinux 8 
    yum update -y
    yum -y install llvm-devel-17.0.6 clang-devel-17.0.6 libzstd-devel zstd ncurses-devel zlib-devel libffi libffi-devel
    export LLVM_DIR=/usr
    export LLVM_SYS_170_PREFIX=/usr
fi