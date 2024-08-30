import json
import os
from typing import List

from etalon.metrics.metric_utils import (
    find_min_tbt_deadline_to_meet,
    get_request_level_deadline_miss_rate,
)
from etalon.metrics.request_metrics import RequestMetrics


class RequestLevelMetrics:
    """
    Array of metrics for all requests
    """

    def __init__(
        self,
        ttft_deadline: float,
        tbt_deadline: float,
        target_deadline_miss_rate: float,
    ) -> None:
        self.ttft_deadline: float = ttft_deadline
        self.tbt_deadline: float = tbt_deadline
        self.target_deadline_miss_rate: float = target_deadline_miss_rate
        self.num_prompt_tokens: List[int] = []
        self.num_output_tokens: List[int] = []
        self.num_total_tokens: List[int] = []
        self.tpot: List[float] = []
        self.ttft: List[float] = []
        self.tbt: List[List[float]] = []
        self.end_to_end_latency: List[float] = []
        self.normalized_end_to_end_latency: List[float] = []
        self.output_throughput: List[float] = []
        self.deadline_miss_rate: List[float] = []
        self.min_tbt_deadline_to_meet: List[float] = []

    def put(self, request_metrics: RequestMetrics):
        self.num_prompt_tokens.append(request_metrics.num_prompt_tokens)
        self.num_output_tokens.append(request_metrics.num_output_tokens)
        self.num_total_tokens.append(request_metrics.num_total_tokens)
        self.tpot.append(request_metrics.tpot)
        self.ttft.append(request_metrics.ttft)
        self.tbt.append(request_metrics.inter_token_times[1:])
        self.end_to_end_latency.append(request_metrics.end_to_end_latency)
        self.normalized_end_to_end_latency.append(
            request_metrics.normalized_end_to_end_latency
        )
        self.output_throughput.append(request_metrics.output_throughput)
        deadline_miss_rate, _, _ = get_request_level_deadline_miss_rate(
            inter_token_times=request_metrics.inter_token_times,
            ttft_deadline=self.ttft_deadline,
            tbt_deadline=self.tbt_deadline,
        )
        self.deadline_miss_rate.append(deadline_miss_rate)
        min_tbt_deadline_to_meet = find_min_tbt_deadline_to_meet(
            inter_token_times=request_metrics.inter_token_times,
            ttft_deadline=self.ttft_deadline,
            target_deadline_miss_rate=self.target_deadline_miss_rate,
        )
        self.min_tbt_deadline_to_meet.append(min_tbt_deadline_to_meet)

    def to_dict(self):
        return {
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_output_tokens": self.num_output_tokens,
            "num_total_tokens": self.num_total_tokens,
            "tpot": self.tpot,
            "ttft": self.ttft,
            "tbt": self.tbt,
            "end_to_end_latency": self.end_to_end_latency,
            "normalized_end_to_end_latency": self.normalized_end_to_end_latency,
            "output_throughput": self.output_throughput,
            "deadline_miss_rate": self.deadline_miss_rate,
            "min_tbt_deadline_to_meet": self.min_tbt_deadline_to_meet,
        }

    def save(self, output_dir: str):
        with open(os.path.join(output_dir, "request_level_metrics.json"), "w") as f:
            json.dump(self.to_dict(), f)
