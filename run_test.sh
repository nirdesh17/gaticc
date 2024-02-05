#!/bin/bash

make -j 8
make libsim -j 8
cd tests
make $1.o -B -j 8 && ./exe/$1 -h
