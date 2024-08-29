from dataclasses import dataclass
from functools import cached_property
from statistics import mean
from typing import List, Optional


@dataclass
class RequestMetrics:
    """
    Request-level metrics for 1 request, all metrics are in seconds.
    """

    inter_token_times: List[float]
    num_prompt_tokens: int
    num_output_tokens: int
    error_msg: Optional[str] = None
    error_code: Optional[int] = None

    @cached_property
    def num_total_tokens(self):
        return self.num_prompt_tokens + self.num_output_tokens

    @cached_property
    def end_to_end_latency(self):
        return sum(self.inter_token_times)

    @cached_property
    def normalized_end_to_end_latency(self):
        if self.num_output_tokens == 0:
            return 0

        return self.end_to_end_latency / self.num_output_tokens

    @cached_property
    def ttft(self):
        if self.num_output_tokens == 0:
            return 0

        return self.inter_token_times[0]

    @cached_property
    def tpot(self):
        if self.num_output_tokens == 0:
            return 0

        if len(self.inter_token_times) < 2:
            return 0

        return mean(self.inter_token_times[1:])

    @cached_property
    def output_throughput(self):
        if self.end_to_end_latency == 0:
            return 0

        return self.num_output_tokens / self.end_to_end_latency
