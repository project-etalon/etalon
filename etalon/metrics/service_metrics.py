import time
from typing import Dict

from etalon.metrics.metric_store import MetricStore
from etalon.metrics.request_metrics import RequestMetrics


class ServiceMetrics:
    def __init__(
        self,
        timeout: float,
        max_requests: int,
        ttft_deadline: float = 0.1,
        tbt_deadline: float = 0.05,
        target_deadline_miss_rate: float = 0.1,
        should_write_metrics: bool = True,
        wandb_project: str = None,
        wandb_group: str = None,
        wandb_run_name: str = None,
    ) -> None:
        self.timeout = timeout
        self.max_requests = max_requests
        self.start_time = None
        self.end_time = None

        self.metric_store = MetricStore(
            timeout=timeout,
            max_requests=max_requests,
            ttft_deadline=ttft_deadline,
            tbt_deadline=tbt_deadline,
            target_deadline_miss_rate=target_deadline_miss_rate,
            should_write_metrics=should_write_metrics,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_run_name=wandb_run_name,
        )

    @property
    def num_requests(self) -> int:
        return self.metric_store.num_requests

    @property
    def num_completed_requests(self) -> int:
        return self.metric_store.num_completed_requests

    @property
    def num_errored_requests(self) -> int:
        return self.metric_store.num_errored_requests

    @property
    def duration(self):
        assert self.end_time is not None
        assert self.start_time is not None

        return self.end_time - self.start_time

    @property
    def completed_requests_per_min(self):
        return self.num_completed_requests / self.duration * 60

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    def should_stop(self):
        return not (
            time.monotonic() - self.start_time < self.timeout
            and self.num_completed_requests < self.max_requests
        )

    def register_launched_request(self):
        self.metric_store.register_launched_request()

    def add_request_metrics(self, request_metrics: RequestMetrics):
        self.metric_store.add_request_metrics(request_metrics)

    def get_duration_summary(self) -> Dict[str, float]:
        return {
            "Duration": self.duration,
            "Completed Requests Per Min": self.completed_requests_per_min,
        }

    def get_aggregated_summary(self) -> Dict[str, float]:
        return {
            **self.metric_store.get_aggregated_summary(),
            **self.get_duration_summary(),
        }

    def get_summary(self) -> Dict[str, float]:
        return {
            **self.metric_store.get_summary(),
            **self.get_duration_summary(),
        }

    def __str__(self) -> str:
        return "\n".join(
            [f"{k}: {v:.5f}" for k, v in self.get_aggregated_summary().items()]
            + [str(summary) for summary in self.metric_store.summaries.values()]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def store_output(self, output_dir: str):
        self.metric_store.store_output(output_dir)
