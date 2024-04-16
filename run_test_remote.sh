#!/bin/bash

# Remote server details
remote_user="shreeyash"
remote_host="192.168.29.9"
remote_folder="/home/$remote_user/sysim"

remote_file="$remote_folder/run_test.sh"

# Check if the argument is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <test_name>"
    exit 1
fi

# Read the test name from the command-line argument
argument="$1"
shift 1

tar -cvzf sysim.tar.gz --exclude='onnx.pb.*' src/*.{cpp,h,py} Makefile tests/*.cpp tests/Makefile *.proto *.sh
ssh "$remote_user@$remote_host" "rm -rf $remote_folder && mkdir -p $remote_folder/ $remote_folder/tests/exe \
  $remote_folder/obj/ 2>/dev/null"
scp sysim.tar.gz  "$remote_user@$remote_host:$remote_folder"
ssh "$remote_user@$remote_host" "cd $remote_folder && \
  tar -xvzf sysim.tar.gz && \ 
  ./build.sh all && \
  $remote_file $argument $@"
