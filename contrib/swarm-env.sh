# Env variables need by Swarm

# EBBRT_NODE_ALLOCATOR_CUSTOM_NETWORK_IP_CMD: indicates the NodeAllocator the ip address of weave network
# EBBRT_NODE_ALLOCATOR_CUSTOM_NETWORK_NODE_CONFIG: indacates the NodeAllocator to run the containers over weave
# EBBRT_NODE_LIST_CMD: indicates the pool of nodes to round robin 
# DOCKER_HOST: indicate the swarm manager host to launch container into the swarm 


export EBBRT_NODE_ALLOCATOR_CUSTOM_NETWORK_IP_CMD="ip addr show weave | grep 'inet ' | cut -d ' ' -f 6"
export EBBRT_NODE_ALLOCATOR_CUSTOM_NETWORK_NODE_CONFIG="--net=weave -e IFACE_DEFAULT=ethwe0"
export EBBRT_NODE_LIST_CMD='echo node194'
export DOCKER_HOST=192.168.1.135:4000
