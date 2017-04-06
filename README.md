# EbbRT_fetalRecon

## Requirements
* Build and install EbbRT native toolchain, assume installed at `~/sysroot/native`
* Build and install EbbRT hosted library, assume installed at `~/sysroot/hosted`

## Working with Docker Swarm
* Install docker and set-up weave this can be done by executing the saltscripts
```
$ salt <minion_id> state.apply docker.swarm
```
Also, create the consul, manager and some workers:
```
$ salt <minion_id> state.apply docker.swarm.consul 
$ salt <minion_id> state.apply docker.swarm.manager
$ salt <minion_id> state.apply docker.swarm.worker 
```

## Environment variables
#### To use the Weave network
```
export EBBRT_NODE_ALLOCATOR_CUSTOM_NETWORK_IP_CMD="ip addr show weave | grep 'inet ' | cut -d ' ' -f 6”
export EBBRT_NODE_ALLOCATOR_CUSTOM_NETWORK_NODE_CONFIG="--net=weave -e IFACE_DEFAULT=ethwe0"
```
#### To round-robin through the physical nodes when using Weave
```
export EBBRT_NODE_LIST_CMD='docker network ls |grep weave | tr -s " " | cut -d " " -f 2 | cut -d "/" -f 1 | paste -sd "," -'
```

## Build
```
EBBRT_SYSROOT=~/sysroot/native CMAKE_PREFIX_PATH=~/sysroot/hosted make -j
```

## Run with small dataset
```
./small.sh <threads> <iterations> <back_end_nodes> <front_end_cpus>
```
Example:
```
./small.sh 2 4 2 3
```

## Run with large dataset
```
./large.sh <threads> <iterations> <back_end_nodes> <front_end_cpus>
```

**Note:** both datasets must be run with *at least* 2 threads.

## Example output
```
./small.sh 2 1 2 1
+ export EBBRT_NODE_ALLOCATOR_DEFAULT_CPUS=2
+ export EBBRT_NODE_ALLOCATOR_DEFAULT_RAM=4
+ export EBBRT_NODE_ALLOCATOR_DEFAULT_NUMANODES=1
+ [ -z  ]
+ DEBUG=0
+ [ -n 1 ]
+ TESTARGS= —recIterationsFirst 1 —recIterationsLast 1
+ ./build/reconstruction -o 3TStackReconstruction.nii -i data/masked_stack-1.nii data/masked_stack-2.nii data/masked_stack-3.nii data/masked_stack-4.nii —disableBiasCorrection —useAutoTemplate —useSINCPSF —resolution 2.0 —debug 0 —numThreads 2 —useCPU —iterations 1 —numNodes 2 —numFrontEndCpus 1 —recIterationsFirst 1 —recIterationsLast 1
Node List: node135,node194
Pool Allocation Details:
|      img: /root/multithread/EbbRT-fetalRecon/./build/bm/reconstruction.elf32
|    nodes: 2
| host ids: node135 node194
|     cpus: 1
Creating custom network
Network Details:
| id:
| ip: 10.32.0.1:60847
Docker Container:
|  img: ebbrt/kvm-qemu:latest
|  cid: b1c60c388598
|  log: docker logs b1c60c388598
Node Allocation Details:
| img: /root/multithread/EbbRT-fetalRecon/./build/bm/reconstruction.elf32
|  id: ebbrt-0.26885.0
|  ip: 10.32.0.4
ERRO[0000] error getting events from daemon: unexpected EOF
Docker Container:
|  img: ebbrt/kvm-qemu:latest
|  cid: 083fe422bfd7
|  log: docker logs 083fe422bfd7
Node Allocation Details:
| img: /root/multithread/EbbRT-fetalRecon/./build/bm/reconstruction.elf32
|  id: ebbrt-0.26885.1
|  ip: 10.44.0.2


[Backend average time] CoeffInit 6.50387
[Backend average time] GaussianReconstruction 0.232275
[Backend average time] SimulateSlices 0.527676
[Backend average time] InitializeRobustStatistics 0.0353555
[Backend average time] EStepI 0.102785
[Backend average time] EStepII 1.65e-05
[Backend average time] EStepIII 4.15e-05
[Backend average time] Scale 0.0226075
[Backend average time] SuperResolution 0.187788
[Backend average time] MStep 0.0222405
[Backend average time] RestoreSliceIntensities 0.004508
[Backend average time] ScaleVolume 0.036641
[Backend average time] SliceToVolumeRegistration 0



[Frontend time] CoeffInit 10.4868
[Frontend time] GaussianReconstruction 4.0779
[Frontend time] SimulateSlices 0.65382
[Frontend time] InitializeRobustStatistics 0.039614
[Frontend time] EStepI 0.106595
[Frontend time] EStepII 0.001734
[Frontend time] EStepIII 0.001371
[Frontend time] Scale 0.024175
[Frontend time] SuperResolution 4.19917
[Frontend time] MStep 0.026088
[Frontend time] RestoreSliceIntensities 0.005347
[Frontend time] ScaleVolume 0.03762
[Frontend time] SliceToVolumeRegistration 0

[Reconstruction loop time] 19.6869
[Initial reconstruction time] 12.7952
[Total time] 42.064

[Cheksum] _reconstructed: 44331204.000000
[Cheksum] _mask: 43801.000000
removing container: 083fe422bfd7
removing container: b1c60c388598
```

To checkout, open another terminal on the node where the container is running and do:
```
docker logs <container_id>
```
