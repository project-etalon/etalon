import glob
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from etalon.logger import init_logger
from etalon.request_generator.request_generator_config import RequestGeneratorConfig
from etalon.run_benchmark import parse_args, run_benchmark

logger = init_logger(__name__)

# Prefill lengths profile over, all powers of 2 between 256 and 128K
PREFILL_VALUES = [2**i for i in range(8, 15)]
# Model to train on the prefill values and prefill times
PREFILL_MODEL = "RandomForestRegressor"
# Random Forest Regressor parameters
PREFILL_RANDOM_FOREST_PARAMS = {
    "n_estimators": 10,
    "random_state": 0,
}
# Request length generator provider for prefill profiling
PREFILL_REQUEST_LENGTH_GENERATOR_PROVIDER = "fixed"
# Polynomial degree for the prefill time predictor
PREFILL_POLYNOMIAL_DEGREE = 2
# RMSE threshold for the prefill time predictor
PREFILL_RMSE_THRESHOLD = 0.05
# Number of Ray clients to use for prefill profiling
PREFILL_NUM_RAY_CLIENTS = 1
# Number of concurrent requests per client for prefill profiling
PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT = 1
# Number of completed requests to wait for before stopping the prefill profiling for a prompt length
PREFILL_MAX_NUM_COMPLETED_REQUESTS = 1


class PrefillProfiler:
    def __init__(self, args) -> None:
        self.args = args
        self.prefill_values = PREFILL_VALUES
        if (
            type(self.args.prefill_lengths) is list
            and len(self.args.prefill_lengths) > 0
        ):
            self.prefill_values = self.args.prefill_lengths
        self.prefill_times = []
        self.model = None
        self.transformer = PolynomialFeatures(
            degree=PREFILL_POLYNOMIAL_DEGREE, include_bias=False
        )

        if PREFILL_MODEL == "RandomForestRegressor":
            self.model = RandomForestRegressor(**PREFILL_RANDOM_FOREST_PARAMS)
        else:
            raise NotImplementedError(f"Model {PREFILL_MODEL} is not implemented")

    def _get_result_file(self, run_dir: str) -> str:
        files = glob.glob(os.path.join(run_dir, f"request_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def run(self):
        request_generator_config = RequestGeneratorConfig(self.args)
        request_generator_config.request_length_generator_provider = (
            PREFILL_REQUEST_LENGTH_GENERATOR_PROVIDER
        )
        for prefill_value in self.prefill_values:
            request_generator_config.fixed_request_generator_prefill_tokens = (
                prefill_value
            )
            run_dir = os.path.join(
                self.args.output_dir, f"{self.args.model}_{prefill_value}"
            )
            os.makedirs(run_dir, exist_ok=True)
            run_benchmark(
                model=self.args.model,
                tokenizer_name=self.args.tokenizer,
                output_dir=run_dir,
                additional_sampling_params=self.args.additional_sampling_params,
                num_ray_clients=PREFILL_NUM_RAY_CLIENTS,
                num_concurrent_requests_per_client=PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT,
                max_num_completed_requests=PREFILL_MAX_NUM_COMPLETED_REQUESTS,
                timeout=self.args.timeout,
                llm_api=self.args.llm_api,
                request_generator_config=request_generator_config,
                should_write_metrics=False,
                wandb_project=self.args.wandb_project,
                wandb_group=self.args.wandb_group,
                wandb_run_name=f"prefill_p{prefill_value}_{self.args.model}",
            )
            if wandb.run:
                wandb.finish()

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
            self.model, os.path.join(self.args.output_dir, "prefill_predictor.pkl")
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
        plt.title(self.args.model)
        plt.legend()
        plt.savefig(os.path.join(self.args.output_dir, "prefill_predictions.png"))

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
        plt.title(self.args.model)
        plt.legend()
        plt.savefig(
            os.path.join(self.args.output_dir, "fine_grained_prefill_predictions.png")
        )

        plt.close()

        if self.args.wandb_project and self.args.should_write_metrics:
            wandb.init(
                project=self.args.wandb_project,
                group=self.args.wandb_group,
                name=f"prefill_profiler_{self.args.model}_{self.args.time_stamp}",
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


if __name__ == "__main__":
    args = parse_args()
    prefill_profiler = PrefillProfiler(args)
    prefill_profiler.run()
