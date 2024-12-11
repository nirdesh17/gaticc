# sysim - Compiler/Simulator/Runtime for Gati DNN accelerator

# Build

## Install Dependencies 

**Arch**:

```
sudo pacman -S python3 python-numpy pkg-config python3-venv
```

**Fedora**:
    
``` 
sudo dnf install python3-devel python3-numpy python3-venv
```

**Ubuntu/Debian**:

```
sudo apt install python3-dev python3-numpy pkg-config python3-venv
```

**Python dependencies**:

Regardless of the OS, these packages have to be installed. If pip on your system
supports installing packages system-wide, run: 

```
pip install -r requirements.txt
```

Otherwise, create a virtual env and install it there:

```
mkdir my_env
cd my_env
python -m venv .
source bin/activate
pip install -r requirements.txt
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
./scripts/install_deps.sh
mkdir build
cmake -B build
cmake --build build
```

TODO: add test instructions here


## Usage

See,
```
sysim -h
```
for usage instructions.

# Versioning

Sysim uses three numbers in the style of <https://semver.org/>:

```
MAJOR.MINOR.PATCH
```

The version number is:

1. Used to track the history of this program
2. **Keep it in sync with the hardware** (i.e. the Gati project)

This is done by assigining a meaning to each number and agreeing with the
hardware maintainers on when to increment which number. 

To keep in sync, the major and minor numbers should always be equal to
that of the hardware. So, if we ask ourselves, which version of the 
hardware is compatible with sysim version 1.3.x, the answer is:
Gati version 1.3.x. Keeping the compatibility intact is in the hands
of the maintainers of both projects. The patch number in both cases
should be the latest available for that `major.minor` combination.

When are version numbers incremented:

- major: when architectures are changed fundamentally. for example, a move
from 9x4x4 SAs to 9x8x8, or 9x8x8 to Mobilenet. 
- minor: when a change is supposed to take place in both hardware and software.
for example, addition of an extra field in some instruction. this requires both
hardware and software to be changed to implemented this feature leading to a 
minor version bump.
- patch: patch numbers are incremented changes agnostic to hardware are made
in the software. for example, when a segfault is patched in the software. as
this demands no change in the hardware, only the patch numbers are increased.

Ideally, a version `a.b.c` is always compatible with the version
`(a-1).(b-1).(c-1)`.

Version bumps are controlled manually by the maintainer (who can push to master)
through the `bump_version.sh` script present in the root of this repo. Read the
script (or `./bump_version.sh` to get usage message) to understand what all it
does. Any push to master should be preceded by a version bump.

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

This guide provides instructions on creating and running test files for various
run functions in the `sysim`. The goal is to understand the process of testing
individual functions.

## Important Files for Writing Tests

1. **executor.cpp**: Implements the logic for executing various neural network
   operations by overriding the Op::LayerBase::run() method for different layer
types. This file contains the execution logic for layers such as convolution,
ReLU, max pooling, and more, handling both floating-point and quantized data
types.

2. **onnx_parser.h**: Defines classes and structures for parsing ONNX models. It
   includes definitions for different layer types, methods for setting
parameters, and executing layers. The file also provides utilities for handling
ONNX model components like initializers, value info, and attributes,
facilitating the conversion of ONNX models into executable graphs.

3. **sim.h**: Contains simulation functionalities for neural network operations.
   This file includes classes and methods for simulating the execution of
layers, handling operations like matrix multiplication, pooling and more. It
also provides profiling tools to measure execution performance.

4. **tensor.h**: Defines tensor data structures and methods for managing
   dimensions and accessing data. This file is essential for handling
multi-dimensional arrays in machine learning models, providing an abstract base
class for tensors and concrete implementations for different tensor types. It
includes methods for tensor operations such as element access, insertion, and
dimension manipulation.

## Understanding and Using Functions

There are multiple functions, such as `run_conv`, `run_relu`, `run_maxpool`,
etc., that simulate different CNN operations. To create a test file for any
function, follow these general steps:

1. **Identify the Function**:  Start by choosing the function you want to test
   (e.g., `run_conv`, `run_relu`). These functions are typically found in the
`executor.cpp` file.

2. **Understand the Function Parameters**:  Look into the source file to
   understand:
   - What inputs the function expects (e.g., tensor data, dimensions,
     configuration parameters).
   - What outputs the function produces (e.g., modified tensors, numerical
     results).
   
   For example, for `run_conv`, check what data format it accepts (like a multi-dimensional array or tensor), the required kernel size, stride, etc.

3. **Map to the Class and Methods**:  If the function is part of a class,
   understand how the class is structured:
   - **Constructor**: How to initialize the class object.
   - **Methods**: Which methods of the class need to be called to properly
     execute the function (e.g., initialization methods, setters for inputs).
   - **Input/Output Parameters**: Identify the required formats for input and
     output.

4. **Create the Test File**:  Once you understand how the function works and its
   requirements, create a test file for it (e.g.,
`functionName_inputDims_outputDims.cpp`). Structure the file to:
   - Initialize the necessary data (e.g., input tensors).
   - Call the function with the appropriate arguments.
   - Capture and check the output to ensure correctness.
   - Include all required headers.

5. **Add Executable in CMakeLists.txt**:  After writing the code, add its
   executable in the `CMakeLists.txt` file of the tests directory as a CTest.
Then, run and check if it's working as required.

6. **Repeat for Other Functions**:  Follow the same steps for each function you
   want to test, such as `run_flatten`, `run_gemm`, `run_quantize_linear`, etc.
Each function may have unique inputs and outputs, so make sure to adjust
accordingly.

### Example Test File 

- Find example test files in the
  [sysim/tests](https://github.com/vicharak-in/sysim/tree/master/tests)
directory. For reference on how to include new tests, check the `CMakeLists.txt`
file located in the same folder. This will help you understand the structure and
integration of new test cases.

## Building and Running Test Files

- By default, **Sysim** builds only the main project without including test files. If you want to include the tests in the build, you need to enable the `BUILD_TESTING` flag during the configuration step. Follow these steps:
  ```
  cmake -B build -DBUILD_TESTING=ON
  cmake --build build
  ```

- After building with tests enabled, you can execute individual test executables:
  - Navigate to the `tests` folder within the `build` directory
  - Run the desired test executable. For example:
    ``` 
    ./test_example 
    ```
- To run all tests together, use `ctest` from the build directory:
  ```
  cd build
  ctest
  ```

> [!NOTE]  
> The `BUILD_TESTING` flag is set to `OFF` by default. If tests are not
> required, you can skip enabling this flag.

## Supported Models

- Sysim supports ONNX models that are hosted on our Galacatos Server. One can
  download these models directly for use with Sysim.

  [Download Supported Models](http://galactos.local:8471/)
