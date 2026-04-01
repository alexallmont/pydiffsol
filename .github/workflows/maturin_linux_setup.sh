#!/usr/bin/env bash
if [ -f "/etc/debian_version" ]; then
    echo Installing LLVM via apt
    apt-get update
    apt-get install -y wget lsb-release software-properties-common gnupg
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    ./llvm.sh 21
    apt-get install -y libpolly-21-dev libzstd-dev zstd libncurses-dev zlib1g-dev
    export LLVM_DIR=/usr/lib/llvm-21
    export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21

    echo Installing SparseSuite via apt
    apt-get install -y libsuitesparse-dev
else
    echo Installing LLVM via yum
    yum update -y
    yum -y install llvm-devel-21.0.6 clang-devel-21.0.6 libzstd-devel zstd ncurses-devel zlib-devel libffi libffi-devel
    export LLVM_DIR=/usr
    export LLVM_SYS_211_PREFIX=/usr

    echo Installing SparseSuite via yum
    yum install -y suitesparse-devel
fi
