# Simulator for CNN

# Build

## Install Dependencies 

### Arch

```
sudo pacman -S boost boost-libs python3 protobuf python-numpy pkg-config
python-numpy
```

### Fedora
    
``` 
sudo dnf install boost-devel python3-devel protobuf-compiler pkg-config
python3-numpy
```

### Ubuntu/Debian

```
sudo apt install libprotobuf{available_version} libprotobuf-dev \
    protobuf-compiler libboost{available_version}-all-dev \
    python{available_version}-dev python{available_version} \
  pkg-config python3-numpy

```

`{available_version}` here is a version number of the packages available in
the apt repos.

### MacOS M1/M2

To install boost and protobuf in mac we can use either [brew](https://brew.sh) or [macports](https://www.macports.org).

**Using Brew**
```
brew install boost protobuf
```

### Python dependencies

Regardless of the OS, these packages have to be installed. If pip on your system
supports installing packages system-wide, run: 

```
pip install numpy jax jaxlib pillow onnx
```

Otherwise, create a virtual env and install it there:

```
mkdir my_env
cd my_env
python -m venv .
source bin/activate
pip install {above_packages}
```

### On board

If install sysim on a board with an FPGA (Vaaman etc.), you would need
additional dependencies that allow CPU-FPGA communication:

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

## Compile

```
cd /path/to/sysim
mkdir build
cd build
cmake ..
cmake --build . 
```

TODO: add test instructions here


## Usage (projected)

See,
```
sysim -h
```
for usage instructions.

# Contributing to sysim

## General 

- Format all your commit messages according to <https://www.conventionalcommits.org/en/v1.0.0/>.
- For bugs, create an issues here: <https://github.com/vicharak-in/sysim/issues/>
- Keep commit message titles succinct. Use the body for further elaboration if
  needed. See <https://cbea.ms/git-commit/>
- PRs should be related to a topic/goal, be easy to review and check for bugs.
  Do not create large PRs with random changes. 
- Write simple and easy to read/maintain code.
- "Pre-mature optimizations are the root of all evil". Measure before you
  optimize. Do not use convoluted features of a language just because you know
  them.
- Read <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines> and what
  and what not to use.
- Use a formatter. sysim uses LLVM style formatting for c++
  <https://clang.llvm.org/docs/ClangFormat.html>
- We do not follow any coding guideline to the T but welcome good style
  suggestions. (suggest through ISSUES)
- You can find some here: <https://google.github.io/styleguide/cppguide.html>
