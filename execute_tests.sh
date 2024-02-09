#!/usr/bin/bash

cd tests/

exe_dir=exe

passed_tests=0
total_tests="$(/usr/bin/find $exe_dir -type f | wc -l)"

while read infile; do
  echo "Running "$infile""
  ./$infile 
  if (( $? == 1 )); then
    passed_tests=$(( passed_tests + 1 ))
  fi
done <<< $(/usr/bin/find $exe_dir -type f)

echo "---------------------------------"
echo "Total Passed Tests: $passed_tests/$total_tests"
