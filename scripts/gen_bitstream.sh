#!/bin/bash
set -e

usage() {
cat <<EOF
Usage:
  ./build_bitstream.sh -r <rah_path> -g <gaticc_path> -m <model.onnx> -f <FPGA>

Options:
  -r   Path to rah project
  -g   Path to gaticc repo
  -m   Path to ONNX model
  -f   FPGA type (example: T120)
  -h   Show this help

Example:
  ./build_bitstream.sh \\
    -r ~/Documents/vicharak/rah \\
    -g ~/Documents/vicharak/gaticc-fork \\
    -m ~/Documents/vicharak/gaticc-fork/onnx/yolov8n_quantized.onnx \\
    -f T120
EOF
}

while getopts "r:g:m:f:uh" opt; do
  case $opt in
    r) RAH_PATH="$OPTARG" ;;
    g) GATICC_PATH="$OPTARG" ;;
    m) MODEL_ONNX="$OPTARG" ;;
    f) FPGA="$OPTARG" ;;
    h) usage; exit 0 ;;
    u) UPDATE_REPO=1 ;;
    *) usage; exit 1 ;;
  esac
done

if [[ -z "$RAH_PATH" || -z "$GATICC_PATH" || -z "$MODEL_ONNX" || -z "$FPGA" ]]; then
  echo "Missing required arguments"
  usage
  exit 1
fi

RAH_PATH=$(realpath "$RAH_PATH")
GATICC_PATH=$(realpath "$GATICC_PATH")
MODEL_ONNX=$(realpath "$MODEL_ONNX")

MODEL_NAME=$(basename "$MODEL_ONNX" .onnx)

echo "RAH    : $RAH_PATH"
echo "GATICC : $GATICC_PATH"
echo "MODEL  : $MODEL_ONNX"
echo "NAME   : $MODEL_NAME"
echo "FPGA   : $FPGA"
echo

cd "$RAH_PATH"

if [[ "$UPDATE_REPO" == "1" ]]; then
  echo "[1/5] Updating rah repo"
  git fetch --all
  git checkout origin/main -f
  git submodule update -f
else
  echo "[1/5] Skipping repo update (use -u to enable)"
fi

echo "[2/5] Running gaticc"
gaticc -g "$MODEL_ONNX" \
      -f "$RAH_PATH/rtl/Gati/src/rtl/common/gen_hardware.vh" \
      --fpga "$FPGA"

echo "[3/5] Disabling auto instantiation"
python "$GATICC_PATH/scripts/disable_auto_inst.py" "$RAH_PATH/rah.xml"

echo "[4/5] Running efx_client"
efx_client -d "$RAH_PATH" -e 2024.2

echo "[5/5] Copying bitstream"

OUT_HEX="$RAH_PATH/outflow/rah.hex"
DEST_DIR="$GATICC_PATH/bitstreams"
DEST_HEX="$DEST_DIR/${MODEL_NAME}_${FPGA}.hex"

mkdir -p "$DEST_DIR"
cp "$OUT_HEX" "$DEST_HEX"

echo
echo "Bitstream ready:"
echo "$DEST_HEX"
