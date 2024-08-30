"""
    This file contains the wrapper for the benchmarking.
    It first launches OPEN AI server for different engines like vllm, lightllm, fastertransformers and sarathi-serve, and then runs -m etalon.run_benchmark
"""

import os
import re
import signal
import socket
import subprocess
import time

import ray
from jinja2 import Environment, FileSystemLoader
from ray.util import get_node_ip_address

from etalon.capacity_search.config.config import BenchmarkConfig, JobConfig
from etalon.capacity_search.ray_utils import ReplicaResourceMapping, get_ip
from etalon.logger import init_logger

logger = init_logger(__name__)


def extract_ip(string):
    return re.findall(r"[0-9]+(?:\.[0-9]+){3}", string)[0]


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@ray.remote
class OpenAIServerWrapper:
    def __init__(
        self, replica_resource_mapping: ReplicaResourceMapping, port: int = None
    ):
        self.process = None
        self.port = port
        self.replica_resource_mapping = replica_resource_mapping
        gpu_devices = []
        curr_node_ip = get_node_ip_address()
        for i in range(len(self.replica_resource_mapping["0"])):
            node_ip = extract_ip(self.replica_resource_mapping["0"][i][0])
            if curr_node_ip == node_ip:
                gpu_devices.append(self.replica_resource_mapping["0"][i][1])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpu_devices])

    def get_openai_server_command(
        self,
        openai_server_engine=None,
        openai_server_model="gpt-3.5-turbo",
        openai_api_key=None,
        tp=1,
        pp=1,
        rope_scaling_type=None,
        rope_scaling_factor=None,
        max_num_batched_tokens=512,
    ) -> str:
        template_dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "engine_templates")
        )
        env = Environment(loader=FileSystemLoader(template_dir_path))
        template = None
        cuda_devices = None
        if openai_server_engine == "vllm":
            assert (
                openai_api_key is not None
            ), "OpenAI API key is required for vLLM engine"
            template = env.get_template("vllm_template.jinja")
        elif openai_server_engine == "sarathi":
            assert (
                openai_api_key is not None
            ), "OpenAI API key is required for Sarathi engine"
            template = env.get_template("sarathi_template.jinja")
        elif openai_server_engine == "tgi":
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            del os.environ["CUDA_VISIBLE_DEVICES"]
            template = env.get_template("tgi_template.jinja")
        else:
            raise ValueError(f"Invalid engine: {openai_server_engine}")

        rope_scaling = (
            f'{{"type":"{rope_scaling_type}", "factor": {rope_scaling_factor}}}'
        )

        data = {
            "openai_server_model": openai_server_model,
            "openai_api_key": openai_api_key,
            "tp": tp,
            "pp": pp,
            "port": self.port,
            "rope_scaling": rope_scaling,
            "rope_scaling_type": rope_scaling_type,
            "rope_scaling_factor": rope_scaling_factor,
            "cuda_devices": cuda_devices,
            "max_num_batched_tokens": max_num_batched_tokens,
        }

        cmd = template.render(data)

        return cmd

    def launch_openai_server(
        self,
        openai_server_engine=None,
        openai_server_model="gpt-3.5-turbo",
        openai_api_key=None,
        tp=1,
        pp=1,
    ):
        """
        Setup the OPEN AI server
        If no engine is specified, it defaults to actual OPEN AI server itself.
        """
        openai_server_command = None
        if openai_server_engine in ["vllm", "sarathi", "tgi"]:
            openai_server_command = self.get_openai_server_command(
                openai_server_engine=openai_server_engine,
                openai_server_model=openai_server_model,
                openai_api_key=openai_api_key,
                tp=tp,
                pp=pp,
                rope_scaling_type=(
                    "linear" if openai_server_engine == "vllm" else "dynamic"
                ),
                rope_scaling_factor=4.0,
                max_num_batched_tokens=512,
            )
            logger.info(
                f"Launching OPEN AI server with command: {openai_server_command}"
            )
            self.process = subprocess.Popen(
                openai_server_command, shell=True, preexec_fn=os.setsid
            )
        elif openai_server_engine == "default" or openai_server_engine is None:
            # just use the actual OPEN AI server
            pass
        else:
            logger.error(f"Invalid engine: {openai_server_engine}")
            raise ValueError(f"Invalid engine: {openai_server_engine}")

    def stop_openai_server(self):
        """
        Stops the OPEN AI server
        """
        if self.process is not None:
            logger.info("Stopping OPEN AI server")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)


def setup_api_environment(
    openai_server_engine=None,
    openai_api_key=None,
    openai_port=None,
):
    # just make sure that OPENAI_API_KEY/BASE doesn't change for other ray tasks when setting for this one.
    # checked by printing, and it doesn't change
    if openai_server_engine == "vllm" or openai_server_engine == "sarathi":
        assert openai_api_key is not None, "OpenAI API key is required for VLLM engine"
        assert openai_port is not None, "OpenAI port is required for VLLM engine"
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_API_BASE"] = f"http://localhost:{openai_port}/v1"


def is_default_engine(engine) -> bool:
    return engine == "default" or engine is None


def run(
    job_config: JobConfig,
    benchmark_config: BenchmarkConfig,
    replica_resource_mapping: ReplicaResourceMapping,
):
    """
    Main function
    """

    openai_port = job_config.server_config.port
    num_gpus = (
        job_config.parallel_config.get_num_gpus()
        if not is_default_engine(job_config.server_config.openai_server_engine)
        else 0
    )

    # If server runs across multiple nodes, then just create set of nodes and pass it to resources parameter for Ray Actor
    set_of_nodes = set(
        replica_resource_mapping["0"][i][0]
        for i in range(len(replica_resource_mapping["0"]))
    )
    resources = {i: 0.001 for i in set_of_nodes}

    openai_server_wrapper = OpenAIServerWrapper.options(
        num_gpus=num_gpus, resources=resources
    ).remote(replica_resource_mapping=replica_resource_mapping, port=openai_port)

    # Launch the OPEN AI server
    ray.get(
        openai_server_wrapper.launch_openai_server.remote(
            openai_server_engine=job_config.server_config.openai_server_engine,
            openai_server_model=job_config.model_config.identifier,
            openai_api_key=job_config.server_config.openai_api_key,
            tp=job_config.parallel_config.tp_dimension,
            pp=job_config.parallel_config.pp_dimension,
        )
    )

    setup_api_environment(
        openai_server_engine=job_config.server_config.openai_server_engine,
        openai_api_key=job_config.server_config.openai_api_key,
        openai_port=openai_port,
    )

    # Wait for the server to start. For 70B model, it takes around 2 minutes to start
    sleep_time = (
        0 if is_default_engine(job_config.server_config.openai_server_engine) else 60
    )
    time.sleep(sleep_time)

    # Additional retry mechanism to check if server is up
    count = 0
    while not is_port_in_use(openai_port):
        logger.info(
            f"Waiting for OPEN AI server to start. Port {openai_port} is not in use"
        )
        time.sleep(60)
        if count > 5:
            raise RuntimeError("OPEN AI server did not start after 6 mins")
        count += 1

    # Run the benchmark
    benchmark_command = f"python -m etalon.run_benchmark {job_config.to_args()} {benchmark_config.to_args()}"
    logger.info(f"Running benchmark with command: {benchmark_command}")
    benchmark_process = subprocess.Popen(benchmark_command, shell=True)
    benchmark_process.wait()

    # Stop the OPEN AI server
    ray.get(openai_server_wrapper.stop_openai_server.remote())
