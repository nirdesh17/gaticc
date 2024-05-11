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
LD_LIBRARY_PATH=/usr/local/lib protoc --cpp_out=${SRC_DIR} onnx.proto
REMOTE_COMPILE_VAR=0

echo "remote compile " ${REMOTE_COMPILE_VAR}
mkdir tests/exe 2>/dev/null
mkdir obj 2>/dev/null
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  make "$target_name" -j "$(nproc --all)" REMOTE_COMPILE=${REMOTE_COMPILE_VAR} -B
elif [[ "$OSTYPE" == "darwin"* ]]; then
  make "$target_name" -j "$(sysctl -n hw.physicalcpu)" REMOTE_COMPILE=${REMOTE_COMPILE_VAR} -B
else
  echo "Unsupported operating system: $OSTYPE"
  exit 1
fi
