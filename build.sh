#!/bin/bash

function usage() {
  cat << EOF
build.sh [target]
  possible targets: all, test, clean
EOF
}
SRC_DIR=src/
target_name=all
if [ -n "$1" ]; then
  target_name="$1"
elif [ "$1" = "-h" ]; then
  usage
fi 
LD_LIBRARY_PATH=/usr/local/lib protoc --cpp_out=${SRC_DIR} onnx.proto
make "$target_name" -j "$(nproc --all)" -B
