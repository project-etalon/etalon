"""
This file contains the wrapper for the benchmarking.
"""

import os
import re
import socket
import subprocess

from etalon.capacity_search.config.config import BenchmarkConfig, JobConfig
from etalon.logger import init_logger

logger = init_logger(__name__)


def extract_ip(string):
    return re.findall(r"[0-9]+(?:\.[0-9]+){3}", string)[0]


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def setup_api_environment(
    openai_api_key=None,
    openai_port=None,
):
    """Set up environment variables for OpenAI API"""
    assert openai_api_key is not None, "OpenAI API key is required"
    assert openai_port is not None, "OpenAI port is required"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = f"http://localhost:{openai_port}/v1"


def run(
    job_config: JobConfig,
    benchmark_config: BenchmarkConfig,
):
    """Main function to run benchmark"""

    setup_api_environment(
        openai_api_key=job_config.server_config.openai_api_key,
        openai_port=job_config.server_config.port,
    )

    benchmark_command = f"python -m etalon.run_benchmark {job_config.to_args()} {benchmark_config.to_args()}"
    logger.info(f"Running benchmark with command: {benchmark_command}")
    benchmark_process = subprocess.Popen(benchmark_command, shell=True)
    benchmark_process.wait()
    logger.info("Benchmark finished")
