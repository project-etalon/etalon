import pandas as pd

from etalon.logger import init_logger
from etalon.request_generator.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)

logger = init_logger(__name__)


class TraceRequestIntervalGenerator(BaseRequestIntervalGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        trace_file = self.config.trace_file
        # load into a pd dataframe
        self.trace_df = pd.read_csv(trace_file)

        self.trace_df["arrival_time"] = pd.to_datetime(self.trace_df["arrival_time"])
        # restrict trace_df to be a subset of rows that have the same date
        self.trace_df = self.trace_df[
            (self.trace_df["arrival_time"] > self.config.start_time)
            & (self.trace_df["arrival_time"] < self.config.end_time)
        ]

        # change back to seconds
        self.trace_df["arrival_time"] = (
            self.trace_df["arrival_time"] - self.trace_df["arrival_time"].min()
        ) // pd.Timedelta("1s")

        # rescale the time to change QPS
        self.trace_df["arrival_time"] = (
            self.trace_df["arrival_time"] * self.config.time_scale_factor
        )

        # compute the inter-request time
        self.trace_df["inter_request_time"] = self.trace_df["arrival_time"].diff()

        self.next_request_idx = 1

        logger.info(
            f"Loaded interval trace file {trace_file} with {len(self.trace_df)} requests"
        )

    def get_next_inter_request_time(self) -> float:
        if self.next_request_idx >= len(self.trace_df):
            return None

        inter_request_time = self.trace_df.iloc[self.next_request_idx][
            "inter_request_time"
        ]
        self.next_request_idx += 1
        return inter_request_time
