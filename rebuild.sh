#!/bin/sh

#cd ~/build/build/hosted 
#make -j | grep error

cd ~/EbbRT 
make -j

cd ~/EbbRT-fetalRecon/build
rm -rf CMakeCache.txt  CMakeFiles/ Makefile
cmake -DCMAKE_PREFIX_PATH=~/EbbRT/hosted/ -DCMAKE_BUILD_TYPE=Release ../
make -j

cd ~/EbbRT-fetalRecon/


