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

### Python dependencies

Regardless of the OS, these packages have to be installed. If pip on your system
supports installing packages system-wide, run: 

```
pip install numpy jax jaxlib pillow
```

Otherwise, create a virtual env and install it there:

```
mkdir my_env
cd my_env
python -m venv .
source bin/activate
pip install {above_packages}
```

## Compile

```
cd path/to/sysim
./build.sh
```

Additionally, to build all the tests.
```
./build.sh test
```

To run all the tests,
```
make run-tests
```

To run only a single test,
```
./run_test.sh <test_name>
```
`test_name` is name of the source file of the test present in tests/ directory
without `.cpp` extension or the path. 

For example,
```
./run_test.sh conv_transformer
./run_test.sh 3x3-3x3
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

# Contributing to sysim

## General 

- Find a TODO list here: <https://github.com/vicharak-in/sysim/issues/15>
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

## Adding tests

- sysim does not use any fancy testing/mocking frameworks. all tests are
standalone executables that are staticallt linked to libsim.a created
on the fly by a Makefile.
- add new tests by adding a `<test_name>.cpp` file to tests/ directory.
- For ideas on what to put in it, see `.cpp` files for other tests.
- The general idea is to test a single feature by comparing its 'computed'
outputs to a known truthful 'expected' output. For example, to test a `square`
function, for a set of inputs {1,2,3,4,5}, expected outputs are {1,4,9,16,25},
and if your `square` function generates the same, the test is considered Passing.
- All tests return either a 0 (for failed test) or 1 (for passed tests)
- The helper function `generate_test` in utils.h can be used for comparison
and generation of a report. 
