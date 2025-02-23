import argparse

from etalon.capacity_search.capacity_search import CapacitySearch
from etalon.capacity_search.config.config import JobConfig
from etalon.logger import init_logger

logger = init_logger(__name__)


def run_search(
    job_config: JobConfig,
    args: argparse.Namespace,
):
    capacity_search = CapacitySearch(
        job_config,
        args,
    )
    return capacity_search.search()


class SearchManager:
    def __init__(
        self,
        args: argparse.Namespace,
        config: dict,
    ):
        self.args = args
        self.config = config

    def run(self):
        job_configs = JobConfig.generate_job_configs(self.config)

        all_results = []
        for job_config in job_configs:
            logger.info(f"Running search for {job_config}")
            result = run_search(
                job_config,
                self.args,
            )
            all_results.append(result)

        return all_results
