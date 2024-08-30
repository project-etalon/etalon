from dataclasses import dataclass

from etalon.request_generator.interval_generator.config import (
    GammaRequestIntervalGeneratorConfig,
    PoissonRequestIntervalGeneratorConfig,
    StaticRequestIntervalGeneratorConfig,
    TraceRequestIntervalGeneratorConfig,
)
from etalon.request_generator.length_generator.config import (
    FixedRequestLengthGeneratorConfig,
    SyntheticRequestLengthGeneratorConfig,
    TraceRequestLengthGeneratorConfig,
    ZipfRequestLengthGeneratorConfig,
)


@dataclass
class RequestGeneratorConfig:
    def __init__(self, args):
        self.args = args
        self.request_interval_generator_provider = (
            args.request_interval_generator_provider
        )
        self.request_length_generator_provider = args.request_length_generator_provider
        self.fixed_request_generator_prefill_tokens = (
            args.fixed_request_generator_prefill_tokens
        )

    def get_request_interval_generator_config(self):
        if self.request_interval_generator_provider == "gamma":
            return self.get_gamma_request_interval_generator_config()
        elif self.request_interval_generator_provider == "poisson":
            return self.get_poisson_request_interval_generator_config()
        elif self.request_interval_generator_provider == "static":
            return self.get_static_request_interval_generator_config()
        elif self.request_interval_generator_provider == "trace":
            return self.get_trace_request_interval_generator_config()
        else:
            raise ValueError(
                f"Unknown request interval generator provider: {self.request_interval_generator_provider}"
            )

    def get_request_length_generator_config(self):
        if self.request_length_generator_provider == "zipf":
            return self.get_zipf_request_length_generator_config()
        elif self.request_length_generator_provider == "uniform":
            return self.get_synthetic_request_length_generator_config()
        elif self.request_length_generator_provider == "trace":
            return self.get_trace_request_length_generator_config()
        elif self.request_length_generator_provider == "fixed":
            return self.get_fixed_request_length_generator_config()
        else:
            raise ValueError(
                f"Unknown request length generator provider: {self.request_length_generator_provider}"
            )

    def get_gamma_request_interval_generator_config(self):
        return GammaRequestIntervalGeneratorConfig(
            cv=self.args.gamma_request_interval_generator_cv,
            qps=self.args.gamma_request_interval_generator_qps,
            seed=self.args.seed,
        )

    def get_poisson_request_interval_generator_config(self):
        return PoissonRequestIntervalGeneratorConfig(
            qps=self.args.poisson_request_interval_generator_qps,
            seed=self.args.seed,
        )

    def get_static_request_interval_generator_config(self):
        return StaticRequestIntervalGeneratorConfig(
            seed=self.args.seed,
        )

    def get_trace_request_interval_generator_config(self):
        return TraceRequestIntervalGeneratorConfig(
            trace_file=self.args.trace_request_interval_generator_trace_file,
            start_time=self.args.trace_request_interval_generator_start_time,
            end_time=self.args.trace_request_interval_generator_end_time,
            time_scale_factor=self.args.trace_request_interval_generator_time_scale_factor,
            seed=self.args.seed,
        )

    def get_fixed_request_length_generator_config(self):
        return FixedRequestLengthGeneratorConfig(
            prefill_tokens=self.fixed_request_generator_prefill_tokens,
            decode_tokens=self.args.fixed_request_generator_decode_tokens,
            seed=self.args.seed,
        )

    def get_synthetic_request_length_generator_config(self):
        return SyntheticRequestLengthGeneratorConfig(
            min_tokens=self.args.synthetic_request_generator_min_tokens,
            max_tokens=self.args.request_generator_max_tokens,
            prefill_to_decode_ratio=self.args.synthetic_request_generator_prefill_to_decode_ratio,
            seed=self.args.seed,
        )

    def get_zipf_request_length_generator_config(self):
        return ZipfRequestLengthGeneratorConfig(
            theta=self.args.zipf_request_length_generator_theta,
            scramble=self.args.zipf_request_length_generator_scramble,
            min_tokens=self.args.synthetic_request_generator_min_tokens,
            max_tokens=self.args.request_generator_max_tokens,
            prefill_to_decode_ratio=self.args.synthetic_request_generator_prefill_to_decode_ratio,
            seed=self.args.seed,
        )

    def get_trace_request_length_generator_config(self):
        return TraceRequestLengthGeneratorConfig(
            trace_file=self.args.trace_request_length_generator_trace_file,
            prefill_scale_factor=self.args.trace_request_length_generator_prefill_scale_factor,
            decode_scale_factor=self.args.trace_request_length_generator_decode_scale_factor,
            max_tokens=self.args.request_generator_max_tokens,
            seed=self.args.seed,
        )
