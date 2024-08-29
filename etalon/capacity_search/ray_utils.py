import socket
import time
from typing import Dict, List, Optional, Tuple

import ray

from etalon.logger import init_logger

logger = init_logger(__name__)


ResourceMapping = List[Tuple[str, int]]  # List of (node_ip, gpu_id)
ReplicaResourceMapping = Dict[str, ResourceMapping]


def get_ip() -> str:
    return socket.gethostbyname(socket.gethostname())


def get_nodes() -> List[str]:
    cluster_resources_keys = list(ray.available_resources().keys())
    ip_addresses = [
        x
        for x in cluster_resources_keys
        if x.startswith("node:") and x != "node:__internal_head__"
    ]
    return ip_addresses


def get_ready_promises(promises):
    incomplete_promises = []
    for promise in promises:
        try:
            ray.get(promise, timeout=0)
        except ray.exceptions.GetTimeoutError:
            incomplete_promises.append(promise)
        except Exception as e:
            logger.error(f"Error in promise: {e}")
    return incomplete_promises


@ray.remote
class ResourceManager:
    def __init__(self):
        self.nodes = get_nodes()
        self.num_nodes = len(self.nodes)
        self.num_total_gpus = ray.available_resources()["GPU"]

        assert self.num_nodes > 0, "No nodes found in the cluster"
        assert self.num_total_gpus > 0, "No GPUs found in the cluster"
        assert (
            self.num_total_gpus % self.num_nodes == 0
        ), f"Number of GPUs ({self.num_total_gpus}) is not divisible by the number of nodes ({self.num_nodes})"

        self.gpus_per_node = int(self.num_total_gpus // self.num_nodes)

        self.gpu_free_map = {node: [True] * self.gpus_per_node for node in self.nodes}
        self.node_free_map = {node: True for node in self.nodes}

    def get_replica_resource_mapping(
        self, num_gpus: int
    ) -> Optional[ReplicaResourceMapping]:
        """
        Assign node and gpu for a job
        Note that right now we only support single replica mapping
        """

        assert (
            num_gpus <= self.num_total_gpus
        ), f"Requested {num_gpus} GPUs, but only {self.num_total_gpus} are present in the cluster"

        is_multi_node = num_gpus > self.gpus_per_node
        if is_multi_node:
            assert (
                num_gpus % self.gpus_per_node == 0
            ), f"Number of GPUs ({num_gpus}) is not divisible by the number of GPUs per node ({self.gpus_per_node})"
            num_nodes = num_gpus // self.gpus_per_node

            num_free_nodes = sum(self.node_free_map.values())
            if num_free_nodes < num_nodes:
                return {}

            resource_mapping = []
            for node in self.nodes:
                if self.node_free_map[node]:
                    self.node_free_map[node] = False
                    for i in range(self.gpus_per_node):
                        self.gpu_free_map[node][i] = False
                        resource_mapping.append((node, i))

                    if len(resource_mapping) == num_gpus:
                        return {"0": resource_mapping}
        else:
            # all GPUs must be allocated on the same node and contiguously
            for node in self.nodes:
                resource_mapping = []
                for gpu_id, is_gpu_free in enumerate(self.gpu_free_map[node]):
                    # we don't want to allocate gpu combinations like 1,2
                    if not resource_mapping and gpu_id % num_gpus != 0:
                        continue

                    if is_gpu_free:
                        resource_mapping.append((node, gpu_id))
                    else:
                        # this ensures that we allocate contiguously
                        resource_mapping = []

                    if len(resource_mapping) == num_gpus:
                        self.node_free_map[node] = False
                        for _, i in resource_mapping:
                            self.gpu_free_map[node][i] = False
                        return {"0": resource_mapping}

        # currently we only support single replica allocation
        return {}

    def release_resources(self, replica_resource_mapping: ReplicaResourceMapping):
        for resource_mapping in replica_resource_mapping.values():
            for node, gpu_id in resource_mapping:
                self.gpu_free_map[node][gpu_id] = True

        for node in self.nodes:
            if all(self.gpu_free_map[node]):
                self.node_free_map[node] = True


class RayParallelRunner:
    def __init__(self):
        self.resource_manager = ResourceManager.remote()

    def map(self, func, job_configs):
        # try to assign a core to each task
        promises = []

        remote_func = ray.remote(func)

        job_configs_with_num_gpus = [
            (job_config, job_config.get_num_gpus()) for job_config in job_configs
        ]
        # this reduces fragmentation
        job_configs_with_num_gpus.sort(key=lambda x: x[1])

        for job_config, num_gpus in job_configs_with_num_gpus:
            replica_resource_mapping = {}
            while not replica_resource_mapping:
                # try to pop the promises so that we can get error messages
                promises = get_ready_promises(promises)

                replica_resource_mapping = ray.get(
                    self.resource_manager.get_replica_resource_mapping.remote(num_gpus)
                )
                time.sleep(0.1)
            # launch the task
            runner_node = replica_resource_mapping["0"][0][
                0
            ]  # replica 0, first worker, node
            promise = remote_func.options(resources={runner_node: 0.001}).remote(
                self.resource_manager, replica_resource_mapping, job_config
            )
            promises.append(promise)

        return ray.get(promises)
