# Gaticc - A Compiler/Simulator/Runtime for Gati CNN accelerator

## Build

### Install Dependencies 

**Arch**:
```
sudo pacman -S python3 python-pip pkg-config cmake
```

**Fedora**:
``` 
sudo dnf install python3-devel python-pip cmake
```

**Ubuntu/Debian**:
```
sudo apt install python3-dev python3 python3-pip pkg-config cmake
```

**MacOs**
```
brew install python pkg-config cmake
```

### On board

On the board, you would need additional dependencies that allow CPU-FPGA communication:

First, check if "Vaaman FPGA communication" is checked in the overlay config,
find a detailed how-to
[here](https://docs.vicharak.in/vaaman-linux/linux-configuration-guide/vicharak-config-tool/#vicharak-config-overlays).
Consider, rebooting after this.

Next, add Vicharak's apt repo to your board's apt config. Follow [this
guide](https://docs.vicharak.in/vicharak_sbcs/vaaman/vaaman-linux/linux-configuration-guide/vicharak-apt-config/)
to do this.

After this is setup, run:

```
sudo apt update && sudo apt upgrade
sudo apt install rah-service
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Compile

```
cd /path/to/gaticc
./scripts/install_deps.sh
mkdir build
cmake -B build
cmake --build build && sudo cmake --install build
pip install -e .
```

Compile with debug flags:
```
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

## Usage

See,
```
gaticc -h
```
for usage instructions.

> [!NOTE]
> If you encounter:
>
> ```text
> gaticc: error while loading shared libraries: libprotobuf-lite.so.32: cannot open shared object file: No such file or directory
> ```
>
> Refresh the system library cache:
>
> ```bash
> sudo ldconfig
> ```
>
> Then try running `gaticc -h` again.

### Python Interface

Here's an example script to run simulation of a model (install model files from the model zoo):
```
import gati

onnx_path = "tests/models/mnist_6_28_int8.onnx"
print(np.argmax(np.squeeze(gati.sim(onnx_path, np.load("mnist_2.py")), axis=1), axis=-1))
```

For more examples, checkout `examples/` directory. It contains, scripts to
compile, run, summarize etc.  

For full api doc, read `python/gati.py` (or feed it to your favorite LLM).
