#!/usr/bin/env bash

cd ./cmp_stack/cmp_c_library/cmake-build-release/
rm -r *
cmake ..
make