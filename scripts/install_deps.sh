#!/usr/bin/bash

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $script_dir/..
git submodule update --init --recursive --depth 1 -j "$(nproc --all)"
cd third_party/boost/
./bootstrap.sh
./b2 headers
