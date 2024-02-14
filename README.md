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

## Compile

```
cd path/to/sysim
make -j $(nproc --all)
```
Additionally,
```
make test -j $(nproc --all)
```


# Todo

- float_sasa_test, same as sasa_test
- float incremental tests
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
- compiler gati-cc
