# EbbRT_fetalRecon
===

## Requirements
* Build and install EbbRT native toolchain, assume installed at `~/sysroot/native`
* Build and install EbbRT hosted library, assume installed at `~/sysroot/hosted`

## Build
```
	EBBRT_SYSROOT=~/sysroot/native CMAKE_PREFIX_PATH=~/sysroot/hosted make -j all
```

## Run with large dataset with 8 threads and 2 iterations

```
   ./large.sh 8 2
```

## Run with small dataset with 8 threads and 2 iterations

```
   ./small.sh 8 2
```

## Example output
```
   ./large.sh 8 2
+ ./build/reconstruction -o 3TReconstruction.nii -i data/14_3T_nody_001.nii data/10_3T_nody_001.nii data/21_3T_nody_001.nii data/23_3T_nody_001.nii -m data/mask_10_3T_brain_smooth.nii --disableBiasCorrection --useAutoTemplate --useSINCPSF --resolution 1.0 --debug 0 --numThreads 8 --useCPU --iterations 2
_max_intensity = 2532.843568, _mmin_intensity = 2.956555
Network Details:
| id: f64fff732999
| ip: 172.20.0.1:34921
# debug w/ wireshark:
# wireshark -i br-f64fff732999 -k
Docker Container:
|  img: ebbrt/kvm-qemu:debug
|  cid: 4e1a6cfa37cc
|  log: docker logs 4e1a6cfa37cc
Node Allocation Details: 
| img: /home/handong/github/ebbrt-contrib/EbbRT_fetalReconstruction/./build/bm/reconstruction.elf32
|  id: ebbrt-191873.32432.0
|  ip: 172.20.0.2
# debug w/ gdb: 
# gdb /home/handong/github/ebbrt-contrib/EbbRT_fetalReconstruction/./build/bm/reconstruction.elf -q -ex 'set tcp connect-timeout 60' -ex 'target remote 172.20.0.2:1234'
all nodes initialized
SendRecon : Sending 38524920 bytes
SendRecon: Blocking ... 
void irtkReconstructionEbb::ReceiveMessage(ebbrt::Messenger::NetworkId, std::unique_ptr<ebbrt::IOBuf>&&), Received 8870840 bytes, 0
Reconstructed sum = 697141504.000000
SendRecon: returned from future
total time: 574.958156 seconds
EBBRT ends
removing container: 4e1a6cfa37cc
removing Network: f64fff732999

```

To checkout, open another terminal and do 
```
  docker logs -f 4e1a6cfa37cc
```


## Note, there is a race somewhere during initial sending of bytes so may hang

