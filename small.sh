#!/bin/sh
set -x
#kh clean
#sudo kill -9 $(ps aux | grep '[d]nsmasq --pid-file=/opt/khpy' | awk '{print $2}')
#sudo kill -9 $(ps aux | grep '[q]emu-system-x86_64' | awk '{print $2}')
export EBBRT_NODE_ALLOCATOR_DEFAULT_CPUS=$1
export EBBRT_NODE_ALLOCATOR_DEFAULT_RAM=4
export EBBRT_NODE_ALLOCATOR_DEFAULT_NUMANODES=1
./build/reconstruction -o 3TStackReconstruction.nii -i data/masked_stack-1.nii data/masked_stack-2.nii data/masked_stack-3.nii  data/masked_stack-4.nii --disableBiasCorrection --useAutoTemplate --useSINCPSF --resolution 2.0 --debug 1 --numThreads $1 --useCPU --iterations $2 --numNodes $3 --numFrontEndCputs $4 
