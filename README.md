# Simulator for CNN

# Build

## Install Dependencies 

### Arch

```
sudo pacman -S boost boost-libs python3 protobuf 
```

### Fedora
    
``` 
sudo dnf install boost-devel python3-devel protobuf-compiler 
```

### Ubuntu/Debian

```
sudo apt install libprotobuf{available_version} libprotobuf-dev \
    protobuf-compiler libboost{available_version}-all-dev \
    python{available_version}-dev python{available_version}
```

`{available_version}` here is a version number of the packages available in
the apt repos.

### Mac OS M1/M2

To install boost and protobuf in mac we can use either [brew](https://brew.sh) or [macports](https://www.macports.org).

**Using Brew**
```
brew install boost protobuf
```

## Compile

```
cd path/to/sysim
make -j $(nproc --all)
```
Additionally,
```
make test -j $(nproc --all)
```

## Usage (projected)

```
# print layers in the model
sysim --onnx model.onnx --summary

# get a time-estimate give an architecture
sysim --onnx model.onnx --timeest 9,8,8

# get layer info
sysim --onnx model.onnx --layer-info 'vgg0_conv2_fwd'

# save/cache outputs/inputs of a layer on disk
sysim --onnx model.onnx --layer-output 'vgg0_conv2_fwd' -o layer_out.save
sysim --onnx model.onnx --layer-input 'vgg0_conv2_fwd' -o layer_in.save

# compare two layer saves
sysim --compare layer.save layer2.save 
```

# Todo

- Float shift bug
    - python interpreter, main program loading the same program
    - load a custom compiled protobuf
- change the ffi make wrap around torch.nn only
- custom shape inference due to stale value_info_protos after quantization
- executor
- memory leaks (valgrind) 
- full const correctness
- im2col
    - multi size kernels
    - strided convo
    - DW/PW convo
    - dilated convo
    - padded convo
- onnx parser test scripts
- onnx optimizer + shape inference

# compiler/rt (speculative)

## compiler

### parser/instruction generator

- parse an onnx model reliably
- perform architecture agnostic optimizations
- generate configuration instructions based on the architecture
    - how to specify an architecture?
    - how to represent configuration?

### architecture designer/verilog generator

- given an onnx model, determine what the best architecture for it is
    - best in latency, resource usage, feasilbility
- glue templated verilog files to create the architecture
- additionally, synthesize and generate bitstreams for it
    - could be local with files queried and stiched from an encrypted-db (like 
      an encrypted IP store (efinix does not support ARM though)
    - or cloud based synthesis

## runtime

- given a bitstream, program the fpga with it
- feed image/video stream to the fpga and receive the results reliably
- provide a way to add preprocess/postprocess on inputs/results (python ffi)
- 
