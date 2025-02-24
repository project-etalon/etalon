import json
import os
from typing import DefaultDict, Dict, Optional

import pandas as pd
import plotly_express as px
import wandb

from etalon.config.config import DeadlineConfig, MetricsConfig, PrefillProfilerConfig
from etalon.logger import init_logger
from etalon.metrics.cdf_sketch import CDFSketch
from etalon.metrics.metric_utils import (
    find_min_tbt_deadline_to_meet,
    get_deadline_miss_rate_for_target_tbt_values,
    get_request_level_deadline_miss_rate,
    get_throughput_metrics,
)
from etalon.metrics.request_level_metrics import RequestLevelMetrics
from etalon.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


TARGET_TBT_RANGE = [i * 0.001 for i in range(1, 101)]
QUANTILE_FOR_DEADLINE_MISS_RATE = 0.99


class MetricStore:
    def __init__(
        self,
        timeout: float,
        max_requests: int,
        deadline_config: DeadlineConfig,
        metrics_config: MetricsConfig,
        prefill_profiler_config: PrefillProfilerConfig,
    ) -> None:
        self.timeout = timeout
        self.max_requests = max_requests

        self.num_requests: int = 0
        self.num_errored_requests: int = 0
        self.num_completed_requests: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error_code_freq: DefaultDict[int, int] = DefaultDict(int)
        self.ttft_deadline = deadline_config.ttft_deadline
        self.tbt_deadline = deadline_config.tbt_deadline
        self.target_deadline_miss_rate = deadline_config.target_deadline_miss_rate
        self.ttft_slack = deadline_config.ttft_slack
        self.service_level_missed_deadlines = 0
        self.service_level_total_deadlines = 0
        self.should_write_metrics = metrics_config.should_write_metrics
        self.wandb_project = metrics_config.wandb_project
        self.wandb_group = metrics_config.wandb_group
        self.wandb_run_name = metrics_config.wandb_run_name

        self.prefill_predictions = prefill_profiler_config.predictions
        self.use_predictions_for_ttft = prefill_profiler_config.use_predictions_for_ttft

        self.request_level_metrics = RequestLevelMetrics(
            deadline_config=deadline_config,
            prefill_profiler_config=prefill_profiler_config,
        )

        self.summaries: Dict[str, CDFSketch] = {
            "num_prompt_tokens": CDFSketch(
                "Number of Prompt Tokens", self.should_write_metrics
            ),
            "num_output_tokens": CDFSketch(
                "Number of Output Tokens", self.should_write_metrics
            ),
            "num_total_tokens": CDFSketch(
                "Number of Total Tokens", self.should_write_metrics
            ),
            "tpot": CDFSketch("Time per Output Token", self.should_write_metrics),
            "ttft": CDFSketch("Time to First Token", self.should_write_metrics),
            "tbt": CDFSketch("Time Between Tokens", self.should_write_metrics),
            "end_to_end_latency": CDFSketch(
                "End to End Latency", self.should_write_metrics
            ),
            "normalized_end_to_end_latency": CDFSketch(
                "Normalized End to End Latency", self.should_write_metrics
            ),
            "output_throughput": CDFSketch(
                "Output Throughput", self.should_write_metrics
            ),
            "deadline_miss_rate": CDFSketch(
                f"Deadline Miss Rate with {self.tbt_deadline}s TBT Deadline, {self.ttft_deadline}s TTFT Deadline, {self.ttft_slack}s TTFT Slack, Using Predictions for TTFT: {self.use_predictions_for_ttft} ",
                self.should_write_metrics,
            ),
            "min_tbt_deadline_to_meet": CDFSketch(
                f"Min Deadline to Meet Target Deadline Miss Rate of {self.target_deadline_miss_rate * 100}%",
                self.should_write_metrics,
            ),
        }

        self._init_wandb()

    def _init_wandb(self):
        if not self.should_write_metrics:
            logger.warn("wandb not initialized")
            return

        wandb.init(
            project=self.wandb_project,
            group=self.wandb_group,
            name=self.wandb_run_name,
            config={
                "timeout": self.timeout,
                "max_requests": self.max_requests,
                "ttft_deadline": self.ttft_deadline,
                "tbt_deadline": self.tbt_deadline,
                "ttft_slack": self.ttft_slack,
                "using_predictions_for_ttft": self.use_predictions_for_ttft,
                "target_deadline_miss_rate": self.target_deadline_miss_rate,
            },
        )
        logger.info("wandb initialized")

    @property
    def error_rate(self):
        return self.num_errored_requests / self.num_requests

    def register_launched_request(self):
        self.num_requests += 1

    def add_request_metrics(self, request_metrics: RequestMetrics):
        if request_metrics.error_code:
            self.error_code_freq[request_metrics.error_code] += 1
            self.num_errored_requests += 1
        else:
            self.num_completed_requests += 1

        for metric_name, cdf_sketch in self.summaries.items():
            ttft_deadline = self.ttft_deadline
            if self.use_predictions_for_ttft:
                assert self.prefill_predictions is not None, "Predictions not found"
                ttft_deadline = (
                    self.prefill_predictions[request_metrics.num_total_tokens]
                    + self.ttft_slack
                )
            if metric_name == "tbt":
                cdf_sketch.extend(request_metrics.inter_token_times[1:])
            elif metric_name == "deadline_miss_rate":
                ttft_deadline = self.ttft_deadline
                (
                    deadline_miss_rate,
                    missed_deadlines,
                    total_deadlines,
                ) = get_request_level_deadline_miss_rate(
                    inter_token_times=request_metrics.inter_token_times,
                    ttft_deadline=ttft_deadline,
                    tbt_deadline=self.tbt_deadline,
                )
                cdf_sketch.put(deadline_miss_rate)
                self.service_level_missed_deadlines += missed_deadlines
                self.service_level_total_deadlines += total_deadlines
            elif metric_name == "min_tbt_deadline_to_meet":
                cdf_sketch.put(
                    find_min_tbt_deadline_to_meet(
                        inter_token_times=request_metrics.inter_token_times,
                        ttft_deadline=ttft_deadline,
                        target_deadline_miss_rate=self.target_deadline_miss_rate,
                    )
                )
            else:
                cdf_sketch.put(getattr(request_metrics, metric_name))

        self.request_level_metrics.put(request_metrics)

    def get_aggregated_summary(self) -> Dict[str, float]:
        return {
            "Number of Requests": self.num_requests,
            "Number of Errored Requests": self.num_errored_requests,
            "Number of Completed Requests": self.num_completed_requests,
            "Error Rate": self.error_rate,
            "Deadline Miss Rate": (
                self.service_level_missed_deadlines / self.service_level_total_deadlines
                if self.service_level_total_deadlines > 0
                else 0
            ),
        }

    def get_summary(self) -> Dict[str, float]:
        perf_summary = {}

        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        return {
            **self.get_aggregated_summary(),
            **perf_summary,
        }

    def store_output(self, output_dir: str):
        perf_csv_path = os.path.join(output_dir, "perf_metrics.csv")
        summary_stats_path = os.path.join(output_dir, "error_stats.json")

        # store request level metrics
        self.request_level_metrics.save(output_dir)

        # store metric objects
        for metric_name, metric_summary in self.summaries.items():
            metric_summary._save_df(metric_summary._to_df(), output_dir, metric_name)
            metric_summary.plot_cdf(output_dir, metric_name, metric_name)

        # store service level deadline stats
        with open(f"{output_dir}/service_level_metrics.json", "w") as f:
            json.dump(
                {
                    "service_level_missed_deadlines": self.service_level_missed_deadlines,
                    "service_level_total_deadlines": self.service_level_total_deadlines,
                    "service_level_deadline_miss_rate": (
                        self.service_level_missed_deadlines
                        / self.service_level_total_deadlines
                        if self.service_level_total_deadlines > 0
                        else 0
                    ),
                },
                f,
            )

        # store performance metrics
        perf_header = self.summaries["num_prompt_tokens"].get_csv_header()
        perf_rows = [perf_header]
        for cdf_sketch in self.summaries.values():
            perf_rows.append(cdf_sketch.to_csv_row())

        with open(perf_csv_path, "w") as f:
            f.write("\n".join(perf_rows))

        # store summary stats
        with open(summary_stats_path, "w") as f:
            json.dump(self.get_summary(), f)

        # store additional outputs
        self.store_additional_outputs(output_dir)

    def store_additional_outputs(self, output_dir: str):
        self.store_deadline_miss_rate_for_target_tbt(output_dir)
        self.store_throughput_metrics(output_dir)
        self.store_ttft_violin_plots(output_dir)
        self.store_generation_stalls(output_dir)

    def store_deadline_miss_rate_for_target_tbt(self, output_dir: str):
        # plot deadline miss rate for target TBT values
        deadline_miss_rate_for_target_tbt_values = (
            get_deadline_miss_rate_for_target_tbt_values(
                tbt_times=self.request_level_metrics.tbt,
                target_tbt_deadline_array=TARGET_TBT_RANGE,
                quantile=QUANTILE_FOR_DEADLINE_MISS_RATE,
            )
        )

        percentile_value = int(QUANTILE_FOR_DEADLINE_MISS_RATE * 100)
        x_axis_label = "Target TBT (ms)"
        y_axis_label = f"Miss Rate P({percentile_value})"

        data = {
            x_axis_label: [int(i * 1e3) for i in TARGET_TBT_RANGE],
            y_axis_label: deadline_miss_rate_for_target_tbt_values,
        }
        df = pd.DataFrame(data)

        with open(
            f"{output_dir}/p{percentile_value}_deadline_miss_rate_for_target_tbt_values.json",
            "w",
        ) as f:
            json.dump(data, f)

        if self.should_write_metrics and wandb.run:
            # plot deadline miss rate for target TBT values
            wandb.log(
                {
                    f"p{percentile_value}_deadline_miss_rate": wandb.plot.line(
                        table=wandb.Table(dataframe=df),
                        x=x_axis_label,
                        y=y_axis_label,
                        title="Deadline Miss Rate for Target TBT Values",
                    )
                },
                step=0,
            )

    def store_throughput_metrics(self, output_dir: str):
        (
            tpot_based_throughput,
            tbt_based_throughput,
            deadline_based_throughput,
        ) = get_throughput_metrics(
            self.request_level_metrics.tpot, self.request_level_metrics.tbt
        )

        throughput_metrics = {
            "tpot_based_throughput": tpot_based_throughput,
            "tbt_based_throughput": tbt_based_throughput,
            "deadline_based_throughput": deadline_based_throughput,
        }

        with open(f"{output_dir}/throughput_metrics.json", "w") as f:
            json.dump(throughput_metrics, f)

        # log plot of throughput metrics to wandb
        data = {
            "Metric Type": ["TPOT Based", "TBT Based", "Deadline Based"],
            "Throughput (tok/s)": [
                tpot_based_throughput,
                tbt_based_throughput,
                deadline_based_throughput,
            ],
        }
        df = pd.DataFrame(data)

        if self.should_write_metrics and wandb.run:
            wandb.log(
                {
                    "throughput_metrics": wandb.plot.bar(
                        table=wandb.Table(dataframe=df),
                        label="Metric Type",
                        value="Throughput (tok/s)",
                        title="Token Throughput",
                    )
                },
                step=0,
            )

    def store_ttft_violin_plots(self, output_dir: str):
        data = {}
        for i, ttft in enumerate(self.request_level_metrics.ttft):
            if str(self.request_level_metrics.num_prompt_tokens[i]) not in data:
                data[str(self.request_level_metrics.num_prompt_tokens[i])] = []
            data[str(self.request_level_metrics.num_prompt_tokens[i])].append(ttft)
        df = pd.DataFrame(
            {
                "ttft": [ttft for ttfts in data.values() for ttft in ttfts],
                "prompt_length": [
                    prompt_length
                    for prompt_length in data.keys()
                    for _ in data[prompt_length]
                ],
            }
        )
        df = df.sort_values("prompt_length", key=lambda x: x.astype(int))
        fig = px.violin(df, x="prompt_length", y="ttft", box=True, points="all")
        fig.update_layout(
            title="TTFT Violin Plot",
            xaxis_title="Number of Prompt Tokens",
            yaxis_title="TTFT (s)",
        )
        fig.write_image(f"{output_dir}/ttft_violin_plot.png")
        if self.should_write_metrics and wandb.run:
            wandb.log({"ttft_violin_plot": fig})
            wandb.log({"ttft_violin_data": wandb.Table(dataframe=df)})

    def store_generation_stalls(self, output_dir: str, request_idx: int = 0):
        # just generate for 1 request for now
        if request_idx >= len(self.request_level_metrics.ttft):
            return
        token_generated_times = [
            self.request_level_metrics.ttft[request_idx]
        ] + self.request_level_metrics.tbt[request_idx]
        for i in range(1, len(token_generated_times)):
            token_generated_times[i] += token_generated_times[i - 1]
        tokens_generated = list(range(len(token_generated_times)))
        data = {
            "Time (s)": token_generated_times,
            "Tokens Generated": tokens_generated,
        }
        fig = px.line(
            data_frame=pd.DataFrame(data),
            x="Time (s)",
            y="Tokens Generated",
            title="Tokens Generated vs Time",
        )
        fig.write_image(f"{output_dir}/tokens_generated_vs_time.png")
        if self.should_write_metrics and wandb.run:
            wandb.log({"tokens_generated_vs_time": fig})
