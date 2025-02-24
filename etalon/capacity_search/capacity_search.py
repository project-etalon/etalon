import argparse
import glob
import json
import os
from typing import Optional, Tuple

import numpy as np
import wandb

from etalon.capacity_search.benchmark_wrapper import run
from etalon.capacity_search.config.config import BenchmarkConfig, JobConfig, _get_hash
from etalon.logger import init_logger

logger = init_logger(__name__)

# Increase upper bound of QPS by this scale during binary search
QPS_INCREASE_SCALE = 2
# Threshold to increase the upper bound of QPS during binary search
VICINITY_THRESHOLD = 0.8


class CapacitySearch:
    def __init__(
        self,
        job_config: JobConfig,
        args: argparse.Namespace,
    ) -> None:
        self.job_config = job_config
        self.args = args

        if (self.args.slo_type == "deadline") and self.args.dynamic_ttft_slo:
            assert (
                self.args.profile_dir is not None
            ), "Deadline SLO needs profiled predictions"

    def _run_benchmark(self, benchmark_config: BenchmarkConfig):
        run(self.job_config, benchmark_config)

    def _get_result_file(self, run_dir: str, metric_name: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"{metric_name}.csv"))
        if len(files) == 0:
            return None

        return files[0]

    def _get_request_level_metrics(self, run_dir: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"request_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def _get_service_level_metrics(self, run_dir: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"service_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def _use_deadline_based_slo(
        self, request_level_metrics_file: str
    ) -> Tuple[bool, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        deadline_miss_rate_array = request_level_metrics["deadline_miss_rate"]

        # Calculate percentile values of deadline miss rate
        deadline_miss_rate = np.quantile(
            deadline_miss_rate_array, self.args.deadline_miss_rate_percentile
        )

        is_under_sla = deadline_miss_rate <= self.args.deadline_miss_rate_slo

        return is_under_sla, deadline_miss_rate

    def _use_tbt_and_ttft_slo(
        self,
        request_level_metrics_file: str,
    ) -> Tuple[bool, float, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        # Get TTFT, TBT request level
        ttft_array = request_level_metrics["ttft"]
        tbt_array = request_level_metrics["tbt"]

        # Merge TBT arrays of each request to make it service level
        combined_tbt_array = []
        for i in range(len(tbt_array)):
            combined_tbt_array.extend(tbt_array[i])

        # Calculate percentile values of TBT, TTFT
        tbt = np.quantile(combined_tbt_array, self.args.tbt_percentile)
        ttft = np.quantile(ttft_array, self.args.ttft_percentile)

        is_under_sla = tbt <= self.args.tbt_slo and ttft <= self.args.ttft_slo

        return is_under_sla, tbt, ttft

    def _use_ttft_and_tpot_slo(
        self,
        request_level_metrics_file: str,
    ) -> Tuple[bool, float, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        # Get TTFT, TPOT at request level
        ttft_array = request_level_metrics["ttft"]
        tpot_array = request_level_metrics["tpot"]

        # Calculate percentile values of TTFT, TPOT
        ttft = np.quantile(ttft_array, self.args.ttft_percentile)
        tpot = np.quantile(tpot_array, self.args.tpot_percentile)

        is_under_sla = ttft <= self.args.ttft_slo and tpot <= self.args.tpot_slo

        return is_under_sla, ttft, tpot

    def _is_under_sla(
        self,
        request_level_metrics_file: str,
        benchmark_config: BenchmarkConfig,
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
        is_under_sla = False
        tbt = None
        ttft = None
        tpot = None
        deadline_miss_rate = None

        if self.args.slo_type == "deadline":
            is_under_sla, deadline_miss_rate = self._use_deadline_based_slo(
                request_level_metrics_file
            )
        elif self.args.slo_type == "tbt_ttft":
            is_under_sla, tbt, ttft = self._use_tbt_and_ttft_slo(
                request_level_metrics_file
            )
        elif self.args.slo_type == "ttft_tpot":
            is_under_sla, ttft, tpot = self._use_ttft_and_tpot_slo(
                request_level_metrics_file
            )
        else:
            raise ValueError(f"Invalid SLO type: {self.args.slo_type}")

        logger.info(
            f"{benchmark_config.to_human_readable_name()}"
            f" - TBT P{self.args.tbt_percentile * 100} Tokens: {tbt}"
            f" - TTFT P{self.args.ttft_percentile * 100} Tokens: {ttft}"
            f" - TPOT P{self.args.tpot_percentile * 100} Requests: {tpot}"
            f" - Deadline Miss Rate P{self.args.deadline_miss_rate_percentile * 100} Requests: {deadline_miss_rate}",
        )
        return (
            is_under_sla,
            tbt,
            ttft,
            tpot,
            deadline_miss_rate,
            benchmark_config.get_run_id(),
        )

    def is_under_sla(
        self, qps: float
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
        job_config_key = self.job_config.get_key()
        slo_key = "tbtslo{}_ttftslo{}_tpotslo{}_ttftslackslo{}_deadlinemissrateslo{}_dynamicttftslo{}".format(
            self.args.tbt_slo,
            self.args.ttft_slo,
            self.args.tpot_slo,
            self.args.ttft_slack_slo,
            self.args.deadline_miss_rate_slo,
            self.args.dynamic_ttft_slo,
        )
        overall_key = "_".join([job_config_key, slo_key])
        # since key is very long, hash it to get a unique key for a particular config
        # just check config.json to know actual config
        hash_key = _get_hash(overall_key)

        benchmark_config = BenchmarkConfig(
            output_dir=os.path.join(
                self.args.output_dir,
                str(self.job_config.server_config.openai_server_engine),
                self.job_config.model_config.name,
                # f"ttft_slack_{self.args.ttft_slack_slo}_tbt_{self.args.tbt_slo}",
                str(self.job_config.request_generator_config.trace_file_name),
                f"{hash_key}_q{qps}",
            ),
            qps=qps,
            tbt_deadline=self.args.tbt_slo,
            ttft_deadline=self.args.ttft_slo,
            ttft_slack=self.args.ttft_slack_slo,
            wandb_project=self.args.wandb_project,
            wandb_group=self.args.wandb_group,
            wandb_run_name=f"qps_{qps}_model_{self.job_config.model_config.name}_engine_{self.job_config.server_config.openai_server_engine}",
            should_write_metrics=self.args.should_write_metrics_to_wandb,
            use_predictions_for_ttft=(self.args.slo_type == "deadline")
            and self.args.dynamic_ttft_slo,
            predictor_dir=self.args.profile_dir,
        )

        run_dir = benchmark_config.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        cached_request_level_metrics_file = self._get_request_level_metrics(run_dir)

        if cached_request_level_metrics_file is not None:
            logger.info(f"Cached results found for {qps}")
            return self._is_under_sla(
                cached_request_level_metrics_file, benchmark_config
            )

        self._run_benchmark(benchmark_config)

        request_level_metrics_file = self._get_request_level_metrics(run_dir)

        assert (
            request_level_metrics_file is not None
        ), f"Service-level metrics file not found for {benchmark_config.to_human_readable_name()}"

        return self._is_under_sla(request_level_metrics_file, benchmark_config)

    def search(self):
        """
        Perform binary search to find the maximum QPS under the SLO
        """

        logger.info(
            f"Starting search for {self.job_config.get_human_readable_name()}",
        )

        left = 0
        right = self.job_config.start_qps * 2
        qps = 0
        last_qps = 0
        max_qps_under_sla = None
        min_qps_over_sla = 2**32

        tbt_at_max_qps = None
        ttft_at_max_qps = None
        tpot_at_max_qps = None
        deadline_miss_rate_at_max_qps = None
        best_run_id = None
        found_valid_qps = False

        for _ in range(self.args.max_iterations):
            logger.info(f"Searching between {left} and {right}")
            # stopping condition - we have reached the minimum granularity
            if abs(left - right) < self.args.min_search_granularity * qps / 100:
                break

            qps = (left + right) / 2
            # round to 2 decimal places
            qps = round(qps, 2)

            if qps == last_qps:
                break

            last_qps = qps

            (
                is_under_sla,
                tbt,
                ttft,
                tpot,
                deadline_miss_rate,
                run_id,
            ) = self.is_under_sla(qps)

            if is_under_sla:
                found_valid_qps = True
                max_qps_under_sla = qps
                tbt_at_max_qps = tbt
                ttft_at_max_qps = ttft
                tpot_at_max_qps = tpot
                deadline_miss_rate_at_max_qps = deadline_miss_rate
                best_run_id = run_id

                if qps > VICINITY_THRESHOLD * right:
                    right = min(right * QPS_INCREASE_SCALE, min_qps_over_sla)

                left = qps
            else:
                right = qps
                min_qps_over_sla = min(min_qps_over_sla, qps)

        if not found_valid_qps:
            logger.info(
                f"No valid QPS found for {self.job_config.get_human_readable_name()}",
            )
            return {}

        logger.info(
            f"Max QPS under SLO for {self.job_config.get_human_readable_name()} - "
            f"QPS: {max_qps_under_sla}, "
            f"TBT P{self.args.tbt_percentile * 100}: {tbt_at_max_qps}, "
            f"TTFT P{self.args.ttft_percentile * 100}: {ttft_at_max_qps}, "
            f"TPOT P{self.args.tpot_percentile * 100}: {tpot_at_max_qps}, "
            f"Deadline Miss Rate P{self.args.deadline_miss_rate_percentile * 100}: {deadline_miss_rate_at_max_qps}"
            f"Best Run ID: {best_run_id}",
        )

        if self.args.wandb_project is not None and self.args.enable_wandb_sweep:
            best_run = wandb.Api().run(f"{self.args.wandb_project}/{best_run_id}")
            best_run.tags.append("BEST_CONFIG")
            best_run.update()

        return {
            **self.job_config.to_config_dict(),
            "max_qps_under_sla": max_qps_under_sla,
            "deadline_miss_rate_at_max_qps": deadline_miss_rate_at_max_qps,
        }
