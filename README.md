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

- existing code clean : adding comments (tree, already marked explain further)
- representation/execution of a neural network(graph) onnx*
- hybrid SA of SA (SASA)
- type strictness (reg_t)
- memory leaks (valgrind) 
- const correctness
- static analysis (coverity)
- fuzzy testing 
- LIT integreation 
- Mat template for floats
- im2col
    - multi size kernels
    - strided convo
    - DW/PW convo
    - dilated convo
    - padded convo
- compiler gati-cc
