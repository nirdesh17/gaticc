# Scripts
> [!WARNING]
> Most scripts in this directory are development utilities that evolved alongside Gati. They were often written for a specific experiment, hardware setup, dataset, or repository layout and may require small modifications before use. If something fails, check paths, dependencies, and script assumptions first. Consider them useful references rather than polished, production-ready tools.

## Runtime Servers

Gati supports running models on remote FPGA hardware over a network. These scripts allow a host machine running Gati to communicate with a deployed FPGA platform and execute inference remotely.

### Overview

```text
Laptop / Workstation
        │
        │ Gati Runtime
        ▼
   TCP Network
        ▼
     Vaaman
        │
        ├── server.py      (RAH transport)
        ├── spi_server.py  (SPI transport)
        └── uart_server.py (UART transport)
```

The host machine performs model execution and dispatches layer requests to the FPGA. The server running on the target platform forwards those requests to the accelerator and returns results back to the host.

---

## server.py

Network bridge using the RAH runtime.

This server communicates directly with the FPGA through the Python RAH bindings (`pyrah`).

#### Features

* Remote model execution
* Multi-layer dispatch support
* Remote bitstream flashing
* FPGA buffer management
* Server reset support

#### Ports

| Port | Purpose               |
| ---- | --------------------- |
| 8080 | Runtime communication |
| 8081 | Bitstream flashing    |
| 9090 | Server control/reset  |

#### Run

```bash
python server.py
```

#### Requirements

* pyrah installed
* RAH driver loaded
* FPGA programmed and accessible

#### Typical Usage

Run on Vaaman:

```bash
python server.py
```

Run Gati on host machine and connect to target IP.

---
> [!WARNING]
> `spi_server.py` and `uart_server.py` were used mainly for debugging and bring-up. They require additional hardware support, pin configuration.. Do not expect them to work out of the box without some setup.
## spi_server.py

Network bridge using SPI communication.

Instead of directly accessing FPGA through RAH, this server forwards packets over an SPI link.

Useful when FPGA accelerator is connected through SPI or when RAH is unavailable.

#### Features

* SPI packet transport
* Multi-layer dispatch
* Remote bitstream flashing
* DWP packet handling
* Async packet reception

#### Default SPI Configuration

```text
SPI Bus      : 1
SPI Device   : 0
SPI Speed    : 8 MHz
Mode         : 0
```

#### Run

```bash
python spi_server.py
```

#### Requirements

```bash
pip install spidev
```

#### Hardware Connections

SPI pin assignments depend on target platform configuration.

Refer to:

```bash
gaticc -h
```

for current SPI pin mapping and configuration details.

#### Typical Usage

```text
Host
  ↓ TCP
spi_server.py
  ↓ SPI
FPGA
```

---

## uart_server.py

Network bridge using UART communication.

Instead of SPI or RAH, packets are forwarded through a serial interface.

Useful during bring-up, debugging, or on platforms where UART is easiest to access.

#### Features

* Remote execution
* Layer dispatch support
* Serial transport
* Bitstream programming support

#### Run

```bash
python uart_server.py
```

#### Typical Usage

```text
Host
  ↓ TCP
uart_server.py
  ↓ UART
FPGA
```

#### Notes

UART provides lower bandwidth than SPI and is primarily intended for debugging, validation, and early platform bring-up.

---

# Selecting a Runtime

| Runtime        | Transport | Recommended Use                     |
| -------------- | --------- | ----------------------------------- |
| server.py      | RAH       | Highest performance, production use |
| spi_server.py  | SPI       | Embedded deployments                |
| uart_server.py | UART      | Debugging and bring-up              |

---

## gen_hardware.sh

Generate Gati hardware configuration from a quantized ONNX model.

This script runs the Gati compiler and updates the hardware configuration for the selected FPGA target.

#### Run

```bash
./gen_hardware.sh \
    -p ~/gati_platform \
    -m model.onnx \
    -f T120
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-p` | Path to Gati platform |
| `-m` | Path to quantized ONNX model |
| `-f` | FPGA target |
| `-h` | Show help |

---

# Development Scripts

| Script | Purpose |
|----------|----------|
| `quantize.py` | Quantize FP32 ONNX models to INT8. |
| `infer.py` | Run ONNX Runtime inference and validate results. |
| `onnx_expose_intermediates.py` | Expose intermediate tensors from an ONNX graph. |
| `onnx_gen.py` | Generate synthetic ONNX models for testing. |
| `ort_sim_cmp.py` | Compare Gati simulator outputs against ONNX Runtime. |
| `extract_images_and_labels.py` | Generate datasets and label files for testing. |
| `im2col.py` | Visualize and debug convolution im2col transformations. |
| `align_app.py` | Web UI for visualizing memory alignment and systolic-array layouts. |
| `build.sh` | Build and install Gati from source. |
| `install_deps.sh` | Install required third-party dependencies. |
| `install_test_models.sh` | Download prebuilt validation models. |
| `nms_setup.sh` | Download datasets and assets for NMS testing. |

