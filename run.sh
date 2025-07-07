#!/bin/bash
set -e

cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local -B build
cmake --build build -j32 
cmake --install build
pip install -e .
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
