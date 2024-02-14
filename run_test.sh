#!/bin/bash

make -j 8 || exit 1
make libsim -j 8 || exit 1
cd tests
make $1.o -B -j 8 && ./exe/$1 $@
