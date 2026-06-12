#!/bin/bash
set -e

usage() {
cat <<EOF
Usage:
  ./gen_hardware.sh -p <gati_platform> -m <model.onnx> -f <FPGA>

Options:
  -p   Path to gati_platform
  -m   Path to ONNX model
  -f   FPGA type (example: T120)
  -h   Show help

Example:
  ./gen_hardware.sh \
      -p ~/gati_platform \
      -m yolov8n_quantized.onnx \
      -f T120
EOF
}

while getopts "p:m:f:h" opt; do
  case $opt in
    p) PLATFORM_PATH="$OPTARG" ;;
    m) MODEL_ONNX="$OPTARG" ;;
    f) FPGA="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [[ -z "$PLATFORM_PATH" || -z "$MODEL_ONNX" || -z "$FPGA" ]]; then
  echo "Missing required arguments"
  usage
  exit 1
fi

PLATFORM_PATH=$(realpath "$PLATFORM_PATH")
MODEL_ONNX=$(realpath "$MODEL_ONNX")

GEN_HW_FILE="$PLATFORM_PATH/hardware/gati/src/rtl/common/gen_hardware.vh"

echo "Platform : $PLATFORM_PATH"
echo "Model    : $MODEL_ONNX"
echo "FPGA     : $FPGA"
echo

echo "[1/1] Generating hardware configuration"

gaticc \
    -g "$MODEL_ONNX" \
    -f "$GEN_HW_FILE" \
    --fpga "$FPGA"

echo
echo "===================================================="
echo "Hardware generation completed successfully."
echo
echo "Generated hardware configuration stored in:"
echo
echo "  $GEN_HW_FILE"
echo
echo "You may now run FPGA synthesis using your preferred flow."
echo "===================================================="
