#!/usr/bin/bash

exe_dir=tests/exe
passed_tests=0
total_tests=0
cd $(pwd)/${exe_dir}
for i in *; do
    echo "Running "$i""
    ./"$i"
    if (( $? == 1 )); then
        passed_tests=$(( passed_tests + 1 ))
    fi
    total_tests=$(( total_tests + 1 ))
done
echo "---------------------------------"
echo "Total Passed Tests: $passed_tests/$total_tests"
