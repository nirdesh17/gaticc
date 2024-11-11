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
pip install numpy pillow onnx serial
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

#
# Guide to Writing Tests

## Overview
This guide provides instructions on creating and running test files for various run functions in the `sysim`. The goal is to understand the process of testing individual functions.

## Important Files for Writing Tests
1. **executor.cpp**: Implements the logic for executing various neural network operations by overriding the Op::LayerBase::run() method for different layer types. This file contains the execution logic for layers such as convolution, ReLU, max pooling, and more, handling both floating-point and quantized data types.

2. **onnx_parser.h**: Defines classes and structures for parsing ONNX models. It includes definitions for different layer types, methods for setting parameters, and executing layers. The file also provides utilities for handling ONNX model components like initializers, value info, and attributes, facilitating the conversion of ONNX models into executable graphs.

3. **sim.h**: Contains simulation functionalities for neural network operations. This file includes classes and methods for simulating the execution of layers, handling operations like matrix multiplication, pooling and more. It also provides profiling tools to measure execution performance.

4. **tensor.h**: Defines tensor data structures and methods for managing dimensions and accessing data. This file is essential for handling multi-dimensional arrays in machine learning models, providing an abstract base class for tensors and concrete implementations for different tensor types. It includes methods for tensor operations such as element access, insertion, and dimension manipulation.

## Understanding and Using Functions
There are multiple functions, such as `run_conv`, `run_relu`, `run_maxpool`, etc., that simulate different CNN operations. To create a test file for any function, follow these general steps:

1. **Identify the Function**:  
   Start by choosing the function you want to test (e.g., `run_conv`, `run_relu`). These functions are typically found in the `executor.cpp` file.

2. **Understand the Function Parameters**:  
   Look into the source file to understand:
   - What inputs the function expects (e.g., tensor data, dimensions, configuration parameters).
   - What outputs the function produces (e.g., modified tensors, numerical results).
   
   For example, for `run_conv`, check what data format it accepts (like a multi-dimensional array or tensor), the required kernel size, stride, etc.

3. **Map to the Class and Methods**:  
   If the function is part of a class, understand how the class is structured:
   - **Constructor**: How to initialize the class object.
   - **Methods**: Which methods of the class need to be called to properly execute the function (e.g., initialization methods, setters for inputs).
   - **Input/Output Parameters**: Identify the required formats for input and output.

4. **Create the Test File**:  
   Once you understand how the function works and its requirements, create a test file for it (e.g., `functionName_inputDims_outputDims.cpp`). Structure the file to:
   - Initialize the necessary data (e.g., input tensors).
   - Call the function with the appropriate arguments.
   - Capture and check the output to ensure correctness.
   - Include all required headers.

5. **Add Executable in CMakeLists.txt**:  
   After writing the code, add its executable in the `CMakeLists.txt` file of the tests directory as a CTest. Then, run and check if it's working as required.

6. **Repeat for Other Functions**:  
   Follow the same steps for each function you want to test, such as `run_flatten`, `run_gemm`, `run_quantize_linear`, etc. Each function may have unique inputs and outputs, so make sure to adjust accordingly.

### Example Test File 
- Find example test files in the [sysim/tests](https://github.com/vicharak-in/sysim/tree/master/tests) directory. For reference on how to include new tests, check the `CMakeLists.txt` file located in the same folder. This will help you understand the structure and integration of new test cases.
## Building and Running Test Files
There are two ways to build and run the test files:

1. **Run Tests via Main Project Build**:
    - Build the Main Project.
    - After building the main project, you can run all test files using the `make test` command.

2. **Build and Run Tests Individually**:
    - Ensure the main project is built.
    - Navigate to the `tests/` directory and build only the tests.
    - Once built, you can run all the test files using `ctest`.
