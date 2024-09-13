#!/bin/bash

function usage() {
  cat << EOF
build.sh [target]
  generates protobuf files and builds sysim
  possible targets: all (default), test, clean
EOF
}
SRC_DIR=src/
target_name=all
if [ -n "$1" ]; then
  target_name="$1"
elif [ "$1" = "-h" ]; then
  usage
fi 

protoc --cpp_out=${SRC_DIR} onnx.proto

# mkdir tests/exe 2>/dev/null
# mkdir obj 2>/dev/null
# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#   make "$target_name" -j "$(nproc --all)" NUMPY_INSTALL_PATH=/usr/lib/python3.12/site-packages/numpy -B 
# elif [[ "$OSTYPE" == "darwin"* ]]; then
#   make "$target_name" -j "$(sysctl -n hw.physicalcpu)" -B
# else
#   echo "Unsupported operating system: $OSTYPE"
#   exit 1
# fi
mkdir build
cd build
cmake ..
