import glob
import json
import multiprocessing
import os
import platform

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from etalon.config.config import (
    BenchmarkConfig,
    FixedRequestLengthGeneratorConfig,
    StaticRequestIntervalGeneratorConfig,
)
from etalon.constants.prefill_constants import *
from etalon.logger import init_logger
from etalon.run_benchmark import run_benchmark

logger = init_logger(__name__)


# RMSE threshold for the prefill time predictor
PREFILL_RMSE_THRESHOLD = 0.05
# Number of Ray clients to use for prefill profiling
PREFILL_NUM_CLIENTS = 1
# Number of concurrent requests per client for prefill profiling
PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT = 1
# Number of completed requests to wait for before stopping the prefill profiling for a prompt length
PREFILL_MAX_NUM_COMPLETED_REQUESTS = 1
# Decode tokens when running the prefill profiler
PREFILL_PROFILER_DECODE_TOKENS = 16
# Prefill lengths profile over, all powers of 2 between 256 and 128K
PREFILL_VALUES = [2**i for i in range(8, 15)]
# Model to train on the prefill values and prefill times
PREFILL_MODEL = "RandomForestRegressor"
# Random Forest Regressor parameters
PREFILL_RANDOM_FOREST_PARAMS = {
    "n_estimators": 10,
    "random_state": 0,
}


class PrefillProfiler:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.prefill_values = PREFILL_VALUES
        if (
            type(self.config.prefill_profiler_config.prefill_lengths) is list
            and len(self.config.prefill_profiler_config.prefill_lengths) > 0
        ):
            self.prefill_values = self.config.prefill_profiler_config.prefill_lengths
        self.prefill_times = []
        self.model = RandomForestRegressor(
            n_estimators=PREFILL_RANDOM_FOREST_PARAMS["n_estimators"],
            random_state=PREFILL_RANDOM_FOREST_PARAMS["random_state"],
        )
        self.transformer = PolynomialFeatures(
            degree=PREFILL_POLYNOMIAL_DEGREE, include_bias=False
        )

        if PREFILL_MODEL != "RandomForestRegressor":
            raise NotImplementedError(f"Model {PREFILL_MODEL} is not implemented")

        # update the config with some fixed constants
        self.config.request_interval_generator_config = (
            StaticRequestIntervalGeneratorConfig()
        )
        self.config.metrics_config.should_write_metrics = False
        self.config.client_config.num_clients = PREFILL_NUM_CLIENTS
        self.config.client_config.num_concurrent_requests_per_client = (
            PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT
        )
        self.config.max_completed_requests = PREFILL_MAX_NUM_COMPLETED_REQUESTS
        self.config.request_length_generator_config = FixedRequestLengthGeneratorConfig(
            decode_tokens=PREFILL_PROFILER_DECODE_TOKENS
        )
        self.base_dir = self.config.metrics_config.output_dir

    def _get_result_file(self, run_dir: str) -> str | None:
        files = glob.glob(os.path.join(run_dir, f"request_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def run(self):
        assert isinstance(
            self.config.request_length_generator_config,
            FixedRequestLengthGeneratorConfig,
        ), "Request length generator must be FixedRequestLengthGeneratorConfig"
        for prefill_value in self.prefill_values:
            self.config.request_length_generator_config.prefill_tokens = prefill_value
            run_dir = os.path.join(
                self.base_dir,
                f"{self.config.client_config.model}_{prefill_value}",
            )
            if os.path.isdir(run_dir):
                logger.info(
                    f"Skipping profiling for prefill value = {prefill_value}..."
                )
            else:
                self.config.metrics_config.wandb_run_name = (
                    f"prefill_p{prefill_value}_{self.config.client_config.model}"
                )
                self.config.metrics_config.output_dir = run_dir
                os.makedirs(run_dir, exist_ok=True)
                logger.info(f"Running profiling for prefill value = {prefill_value}...")
                run_benchmark(self.config)
                logger.info(f"Run benchmark done")
                if wandb.run:
                    wandb.finish()

            logger.info(f"Profiling for prefill value = {prefill_value} done")
            logger.info(f"Analyzing the results for prefill value = {prefill_value}...")
            json_file = self._get_result_file(run_dir)
            if json_file is not None:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    ttft = data["ttft"]
                    logger.info(
                        f"""
                        Prefill value: {prefill_value}, Request level ttfts: {ttft}
                        Mean ttft: {sum(ttft) / len(ttft)}
                        Std ttft: {sum((x - sum(ttft) / len(ttft)) ** 2 for x in ttft) / len(ttft)}
                        Max ttft: {max(ttft)}
                        Min ttft: {min(ttft)}
                        """
                    )
                    self.prefill_times.append(min(ttft))
            else:
                logger.error(
                    f"Could not find the result file {json_file} for {run_dir}"
                )
                exit()
            logger.info(f"Going to the next prefill value")

        transformed_prefill_values = self.transformer.fit_transform(
            np.array(self.prefill_values).reshape(-1, 1)
        )

        self.model.fit(transformed_prefill_values, np.array(self.prefill_times))
        rmse = np.sqrt(
            np.mean(
                (
                    self.model.predict(transformed_prefill_values)
                    - np.array(self.prefill_times)
                )
                ** 2
            )
        )
        logger.info(
            f"Model fitted with prefill values and times with root mean squared error: {rmse}",
        )

        joblib.dump(
            self.model,
            os.path.join(self.base_dir, "prefill_predictor.pkl"),
        )

        # also plot the curve containing model's predictions and actual outputs, and dump it
        plt.plot(self.prefill_values, self.prefill_times, label="Actual")
        plt.plot(
            self.prefill_values,
            self.model.predict(transformed_prefill_values),
            label="Predicted",
        )
        plt.xlabel("Prompt Length")
        plt.ylabel("Prefill Time")
        plt.title(self.config.client_config.model)
        plt.legend()
        plt.savefig(os.path.join(self.base_dir, "prefill_predictions.png"))

        # also do fine-grained plotting
        fine_grained_prefill_values = np.linspace(
            min(self.prefill_values), max(self.prefill_values), 1000
        )
        fine_grained_transformed_prefill_values = self.transformer.fit_transform(
            fine_grained_prefill_values.reshape(-1, 1)
        )
        fine_grained_prefill_times = self.model.predict(
            fine_grained_transformed_prefill_values
        )
        plt.plot(
            fine_grained_prefill_values,
            fine_grained_prefill_times,
            label="Fine-grained Prediction",
        )
        plt.xlabel("Prompt Length")
        plt.ylabel("Prefill Time")
        plt.title(self.config.client_config.model)
        plt.legend()
        plt.savefig(
            os.path.join(
                self.base_dir,
                "fine_grained_prefill_predictions.png",
            )
        )

        plt.close()

        if (
            self.config.metrics_config.wandb_project
            and self.config.metrics_config.should_write_metrics
        ):
            wandb.init(
                project=self.config.metrics_config.wandb_project,
                group=self.config.metrics_config.wandb_group,
                name=f"prefill_profiler_{self.config.client_config.model}_{self.config.timestamp}",
            )
            data = {
                "prefill_lengths": self.prefill_values,
                "prefill_times": self.prefill_times,
            }
            wandb.log(
                {
                    "prefill_times_vs_length": wandb.plot.line(
                        table=wandb.Table(data=pd.DataFrame(data)),
                        x="prefill_lengths",
                        y="prefill_times",
                        title="Prefill Times vs Prefill Lengths",
                    )
                },
                step=0,
            )
            data = {
                "prefill_lengths": fine_grained_prefill_values,
                "predicted_prefill_times": fine_grained_prefill_times,
            }
            wandb.log(
                {
                    "predicted_prefill_times_vs_length": wandb.plot.line(
                        table=wandb.Table(data=pd.DataFrame(data)),
                        x="prefill_lengths",
                        y="predicted_prefill_times",
                        title="Predicted Prefill Times vs Prefill Lengths",
                    )
                },
                step=0,
            )

        # assert rmse < PREFILL_RMSE_THRESHOLD, "Model's RMSE is too high, consider changing the model or the data"

        if self.config.prefill_profiler_config.cache_predictions:
            predictions = {}

            x = np.arange(
                self.config.prefill_profiler_config.max_prefill_tokens_to_predict + 1
            )
            x = x.reshape(-1, 1)
            x_poly = self.transformer.fit_transform(x)
            y = self.model.predict(x_poly)
            for i in range(len(x)):
                predictions[int(x[i][0])] = y[i]

            joblib.dump(
                predictions,
                os.path.join(self.base_dir, "prefill_predictions.pkl"),
            )


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    prefill_profiler = PrefillProfiler(config)
    prefill_profiler.run()
