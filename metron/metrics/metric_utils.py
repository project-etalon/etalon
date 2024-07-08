from typing import List, Tuple

import numpy as np

TBT_QUANTILE_FOR_THROUGHPUT = 0.99
TARGET_DEADLINE_MISS_RATE_FOR_THROUGHPUT = 0.1


def get_request_level_deadline_miss_rate(
    inter_token_times: List[float],
    ttft_deadline: float,
    tbt_deadline: float,
    should_ignore_first_token: bool = False,
) -> Tuple[float, int, int]:
    # calculate the deadline miss rate for a given deadline between tokens
    total_deadlines = 0
    missed_deadlines = 0
    deadline_slack = 0
    curr_missed_deadlines = 0

    for i, inter_token_time in enumerate(inter_token_times):
        if i == 0:
            if should_ignore_first_token:
                continue
            # treat first token specially
            if inter_token_time <= ttft_deadline + deadline_slack:
                deadline_slack += ttft_deadline - inter_token_time
                total_deadlines += 1
                continue
            curr_missed_deadlines = (
                1 + (inter_token_time - deadline_slack - ttft_deadline) // tbt_deadline
            )
        else:
            if inter_token_time <= tbt_deadline + deadline_slack:
                deadline_slack += tbt_deadline - inter_token_time
                total_deadlines += 1
                continue
            curr_missed_deadlines = (inter_token_time - deadline_slack) // tbt_deadline
        missed_deadlines += curr_missed_deadlines
        total_deadlines += curr_missed_deadlines
        # reset as we are starting new deadlines for subsequent tokens
        deadline_slack = 0

    if total_deadlines == 0:
        return 0, 0, 0

    return missed_deadlines / total_deadlines, missed_deadlines, total_deadlines


def get_service_level_deadline_miss_rate(
    request_level_inter_token_times: List[List[float]],
    ttft_deadline: List[float],
    tbt_deadline: List[float],
) -> Tuple[float, int, int]:
    service_level_total_deadlines = 0
    service_level_missed_deadlines = 0
    for i, inter_token_times in enumerate(request_level_inter_token_times):
        missed_deadlines, total_deadlines = get_request_level_deadline_miss_rate(
            inter_token_times, ttft_deadline[i], tbt_deadline[i]
        )[1:]
        service_level_total_deadlines += total_deadlines
        service_level_missed_deadlines += missed_deadlines
    if service_level_total_deadlines == 0:
        return 0, 0, 0
    return (
        service_level_missed_deadlines / service_level_total_deadlines,
        service_level_missed_deadlines,
        service_level_total_deadlines,
    )


def find_min_tbt_deadline_to_meet(
    inter_token_times: List[float],
    target_deadline_miss_rate: float,
    ttft_deadline: float,
    should_ignore_first_token: bool = False,
):
    # find the minimum deadline that meets the target miss rate
    deadline = 1e10
    left = 0
    right = 1e10
    mid = 0
    search_granularity = 1e-4
    while right - left > search_granularity:
        mid = (left + right) / 2
        curr_miss_rate, _, _ = get_request_level_deadline_miss_rate(
            inter_token_times,
            ttft_deadline=ttft_deadline,
            tbt_deadline=mid,
            should_ignore_first_token=should_ignore_first_token,
        )
        if curr_miss_rate > target_deadline_miss_rate:
            left = mid + search_granularity
        else:
            deadline = mid
            right = mid - search_granularity

    return deadline


def get_deadline_miss_rate_for_target_tbt_values(
    tbt_times: List[List[float]],
    target_tbt_deadline_array: List[float],
    quantile: float = 0.99,
) -> List[float]:
    assert len(tbt_times)
    num_requests = len(tbt_times)
    quantile_based_miss_rate = []
    for tbt_deadline in target_tbt_deadline_array:
        deadline_miss_rate = []
        for i in range(num_requests):
            deadline_miss_rate.append(
                get_request_level_deadline_miss_rate(
                    inter_token_times=[0] + tbt_times[i],
                    ttft_deadline=0,
                    tbt_deadline=tbt_deadline,
                    should_ignore_first_token=True,
                )[0]
            )
        quantile_based_miss_rate.append(np.quantile(deadline_miss_rate, quantile))
    return quantile_based_miss_rate


def get_throughput_metrics(
    tpot_times: List[float],
    tbt_times: List[List[float]],
) -> Tuple[float, float, float]:
    assert len(tpot_times) == len(tbt_times)
    num_requests = len(tpot_times)
    mean_tpot = np.mean(tpot_times)
    tbt_times_flattened = []
    for tbt_time in tbt_times:
        tbt_times_flattened.extend(tbt_time)
    p99_tbt = np.quantile(tbt_times_flattened, TBT_QUANTILE_FOR_THROUGHPUT)
    tbt_slo = []
    for i in range(num_requests):
        tbt_slo.append(
            find_min_tbt_deadline_to_meet(
                inter_token_times=[0] + tbt_times[i],
                target_deadline_miss_rate=TARGET_DEADLINE_MISS_RATE_FOR_THROUGHPUT,
                ttft_deadline=0,
                should_ignore_first_token=True,
            )
        )
    tbt_slo = np.array(tbt_slo)
    p99_tbt_slo = np.quantile(tbt_slo, TBT_QUANTILE_FOR_THROUGHPUT)

    tpot_based_throughput = 1 / mean_tpot
    tbt_based_throughput = 1 / p99_tbt
    deadline_based_throughput = 1 / p99_tbt_slo

    return tpot_based_throughput, tbt_based_throughput, deadline_based_throughput
