#!/usr/bin/bash

set -eu

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $script_dir/..
git submodule update --init --recursive --depth 1

cd third_party
# install boost
boost_ver1="1.86.0"
boost_ver2="$(echo $boost_ver1 | sed 's/\./_/g')"
tar_file="boost_$boost_ver2.tar.gz"
curl "https://archives.boost.io/release/$boost_ver1/source/$tar_file" > boost.tar.gz
tar -xzf boost.tar.gz
mkdir -p boost
mv -f boost_"$boost_ver2"/* boost/
rm -rf boost_"$boost_ver2" *.tar.gz
