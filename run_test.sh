#!/bin/bash


make -j 8  || exit 1
make libsim -j 8  || exit 1
cd tests
make $1.o -B -j 8  || exit 1
#if [[ $HOST == "galactos" ]]; then
#  source ~/python/bin/activate
#fi
./exe/$1 $@
