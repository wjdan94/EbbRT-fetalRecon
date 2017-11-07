#!/bin/sh
set -x
if [ -z "${DEBUG}" ]
then
  DEBUG=0
else
  DEBUG=1
fi

if [ -n "${TEST}" ]
then
  TESTARGS=" --recIterationsFirst 1 --recIterationsLast 1"
fi

export EBBRT_NODE_ALLOCATOR_DEFAULT_CPUS=$1
export EBBRT_NODE_ALLOCATOR_DEFAULT_RAM=4
export EBBRT_NODE_ALLOCATOR_DEFAULT_NUMANODES=1

./build/reconstruction -o 3TStackReconstruction.nii -i data/masked_stack-1.nii data/masked_stack-2.nii data/masked_stack-3.nii  data/masked_stack-4.nii --disableBiasCorrection --useAutoTemplate --useSINCPSF --resolution 2.0 --debug ${DEBUG} --numThreads $1 --useCPU --iterations $2 --numNodes $3 --numFrontEndCpus $4 ${TESTARGS} | tee tmp

~/EbbRT/contrib/clean_running_apps.sh
