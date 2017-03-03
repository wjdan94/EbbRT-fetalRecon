#!/bin/sh

set -x
#kh clean

#./hosted/build/Debug/reconstruction -o 3TReconstruction.nii -i data/14_3T_nody_001.nii data/10_3T_nody_001.nii data/21_3T_nody_001.nii data/23_3T_nody_001.nii -m data/mask_10_3T_brain_smooth.nii --disableBiasCorrection --useAutoTemplate --useSINCPSF --resolution 1.0 --debug 0 --numThreads $1 --useCPU --iterations $2
export EBBRT_NODE_ALLOCATOR_DEFAULT_CPUS=$1
export EBBRT_NODE_ALLOCATOR_DEFAULT_RAM=32
export EBBRT_NODE_ALLOCATOR_DEFAULT_NUMANODES=1
./build/reconstruction -o 3TReconstruction.nii -i data/14_3T_nody_001.nii data/10_3T_nody_001.nii data/21_3T_nody_001.nii data/23_3T_nody_001.nii -m data/mask_10_3T_brain_smooth.nii --disableBiasCorrection --useAutoTemplate --useSINCPSF --resolution 1.0 --debug 1 --numThreads $1 --useCPU --iterations $2 --numNodes $3 --numFrontEndCpus $4 
