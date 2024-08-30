import hashlib
import os
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional


def _get_hash(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


@dataclass
class ServerConfig:
    openai_server_engine: Optional[str] = None
    openai_api_key: Optional[str] = None
    port: Optional[int] = 8000

    def get_key(self):
        return f"{self.openai_server_engine}_{self.openai_api_key}_{self.port}"

    def get_human_readable_name(self):
        return f"Server engine: {self.openai_server_engine}"

    def to_config_dict(self):
        return {
            "openai_server_engine": self.openai_server_engine,
            "openai_api_key": self.openai_api_key,
            "port": self.port,
        }


@dataclass
class ModelConfig:
    name: str
    identifier: str
    tokenizer: str = None
    parallel_specs: List[str] = field(default_factory=list)
    traces: List[str] = field(default_factory=list)

    def get_key(self):
        return f"{self.name}"

    def get_human_readable_name(self):
        return f"Model: {self.name}, Tokenizer: {self.tokenizer}"

    def to_config_dict(self):
        return {"model_name": self.identifier, "tokenizer_name": self.tokenizer}

    def to_args(self):
        command = f"--model {self.identifier}"
        if self.tokenizer:
            command += f" --tokenizer {self.tokenizer}"
        return command

    def is_parallel_spec_valid(self, spec_name: str) -> bool:
        return not self.parallel_specs or spec_name in self.parallel_specs

    def is_trace_valid(self, trace_name: str) -> bool:
        return not self.traces or trace_name in self.traces


@dataclass
class ParallelConfig:
    name: str
    tp_dimension: int
    pp_dimension: int

    def get_key(self):
        return f"tp{self.tp_dimension}_pp{self.pp_dimension}"

    def get_human_readable_name(self):
        return f"TP: {self.tp_dimension}, PP: {self.pp_dimension}"

    def get_num_gpus(self):
        return self.tp_dimension * self.pp_dimension

    def to_config_dict(self):
        return {
            "model_tensor_parallel_degree": self.tp_dimension,
            "model_pipeline_parallel_degree": self.pp_dimension,
        }


@dataclass
class RequestGeneratorConfig:
    start_qps: float
    request_interval_generator_provider: str
    request_length_generator_provider: str
    gamma_request_interval_generator_cv: Optional[float] = None
    trace_request_interval_generator_trace_file: Optional[str] = None
    trace_request_interval_generator_start_time: Optional[str] = None
    trace_request_interval_generator_end_time: Optional[str] = None
    trace_request_interval_generator_time_scale_factor: Optional[float] = None
    trace_request_length_generator_trace_file: Optional[str] = None
    trace_request_length_generator_prefill_scale_factor: Optional[float] = None
    trace_request_length_generator_decode_scale_factor: Optional[float] = None
    fixed_request_generator_prefill_tokens: Optional[int] = None
    fixed_request_generator_decode_tokens: Optional[int] = None
    synthetic_request_generator_min_tokens: Optional[int] = None
    synthetic_request_generator_prefill_to_decode_ratio: Optional[float] = None
    zipf_request_length_generator_theta: Optional[float] = None
    zipf_request_length_generator_scramble: Optional[bool] = None
    seed: Optional[int] = 42
    trace_file_name: Optional[str] = None

    def get_key(self):
        key = f"{self.request_interval_generator_provider}_{self.request_length_generator_provider}_{self.start_qps}"
        if self.request_interval_generator_provider == "gamma":
            key += f"_{self.gamma_request_interval_generator_cv}"
        return key

    def get_human_readable_name(self):
        return f"Start QPS: {self.start_qps}, Request interval generator: {self.request_interval_generator_provider}, Request length generator: {self.request_length_generator_provider}"

    def to_config_dict(self):
        config_dict = {
            "request-interval-generator-provider": self.request_interval_generator_provider,
            "request-length-generator-provider": self.request_length_generator_provider,
            "seed": self.seed,
        }
        if self.request_interval_generator_provider == "gamma":
            config_dict["gamma-request-interval-generator-cv"] = (
                self.gamma_request_interval_generator_cv
            )
        elif self.request_interval_generator_provider == "trace":
            config_dict["trace-request-interval-generator-trace-file"] = (
                self.trace_request_interval_generator_trace_file
            )
            config_dict["trace-request-interval-generator-start-time"] = (
                self.trace_request_interval_generator_start_time
            )
            config_dict["trace-request-interval-generator-end-time"] = (
                self.trace_request_interval_generator_end_time
            )
            config_dict["trace-request-interval-generator-time-scale-factor"] = (
                self.trace_request_interval_generator_time_scale_factor
            )

        if self.request_length_generator_provider == "trace":
            config_dict["trace-request-length-generator-trace-file"] = (
                self.trace_request_length_generator_trace_file
            )
            config_dict["trace-request-length-generator-prefill-scale-factor"] = (
                self.trace_request_length_generator_prefill_scale_factor
            )
            config_dict["trace-request-length-generator-decode-scale-factor"] = (
                self.trace_request_length_generator_decode_scale_factor
            )
        elif self.request_length_generator_provider == "fixed":
            config_dict["fixed-request-generator-prefill-tokens"] = (
                self.fixed_request_generator_prefill_tokens
            )
            config_dict["fixed-request-generator-decode-tokens"] = (
                self.fixed_request_generator_decode_tokens
            )
        elif self.request_length_generator_provider == "synthetic":
            config_dict["synthetic-request-generator-min-tokens"] = (
                self.synthetic_request_generator_min_tokens
            )
            config_dict["synthetic-request-generator-prefill-to-decode-ratio"] = (
                self.synthetic_request_generator_prefill_to_decode_ratio
            )
        elif self.request_length_generator_provider == "zipf":
            config_dict["zipf-request-length-generator-theta"] = (
                self.zipf_request_length_generator_theta
            )
            config_dict["zipf-request-length-generator-scramble"] = (
                self.zipf_request_length_generator_scramble
            )
        return config_dict

    def to_args(self):
        config_dict = self.to_config_dict()
        args = []
        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    args.append(f"--{key}")
                else:
                    args.append(f"--{key} {value}")
        return " ".join(args)


@dataclass
class RequestConfig:
    num_ray_clients: Optional[int] = None
    num_concurrent_requests_per_client: Optional[int] = None
    timeout: Optional[int] = None
    max_num_completed_requests: Optional[int] = None
    additional_sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    request_generator_max_tokens: Optional[int] = None

    def to_config_dict(self):
        return {
            "num-ray-clients": self.num_ray_clients,
            "num-concurrent-requests-per-client": self.num_concurrent_requests_per_client,
            "timeout": self.timeout,
            "max-num-completed-requests": self.max_num_completed_requests,
            "additional-sampling-params": self.additional_sampling_params,
            "llm-api": self.llm_api,
            "request-generator-max-tokens": self.request_generator_max_tokens,
        }

    def to_args(self):
        config_dict = self.to_config_dict()
        args = []
        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    args.append(f"--{key}")
                else:
                    args.append(f"--{key} {value}")
        return " ".join(args)

    def get_key(self):
        return f"{self.num_ray_clients}_{self.timeout}_{self.max_num_completed_requests}_{self.llm_api}"

    def to_human_readable_name(self):
        return f"Num ray clients: {self.num_ray_clients}, Num concurrent requests per client: {self.num_concurrent_requests_per_client}, Timeout: {self.timeout}, Max num completed requests: {self.max_num_completed_requests}, LLM API: {self.llm_api}"


@dataclass
class JobConfig:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        request_generator_config: RequestGeneratorConfig,
        request_config: RequestConfig,
        server_config: ServerConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.request_generator_config = request_generator_config
        self.request_config = request_config
        self.server_config = server_config

        self.start_qps = self.request_generator_config.start_qps

    def get_key(self):
        config_keys = [
            self.model_config.get_key(),
            self.parallel_config.get_key(),
            self.request_generator_config.get_key(),
            self.request_config.get_key(),
            self.server_config.get_key(),
        ]

        return "_".join(config_keys)

    def get_human_readable_name(self):
        substrings = [
            self.model_config.get_human_readable_name(),
            self.parallel_config.get_human_readable_name(),
            self.request_generator_config.get_human_readable_name(),
            self.request_config.to_human_readable_name(),
            self.server_config.get_human_readable_name(),
            f"Hash: {_get_hash(self.get_key())}",
        ]
        return ", ".join(substrings)

    def get_num_gpus(self):
        return self.parallel_config.get_num_gpus()

    def to_config_dict(self):
        return {
            **self.model_config.to_config_dict(),
            **self.parallel_config.to_config_dict(),
            **self.request_generator_config.to_config_dict(),
            **self.request_config.to_config_dict(),
            **self.server_config.to_config_dict(),
        }

    def to_args(self):
        model_args = self.model_config.to_args()
        request_generator_args = self.request_generator_config.to_args()
        request_args = self.request_config.to_args()
        return f"{model_args} {request_generator_args} {request_args}"

    @classmethod
    def generate_job_configs(cls, config: dict):
        job_configs = []
        port = 8000
        for (
            model_config,
            parallel_config,
            request_generator_config,
            request_config,
            server_config,
        ) in product(
            config["models"],
            config["parallel_specs"],
            config["request_generator_configs"],
            config["request_configs"],
            config["servers"],
        ):
            model_config = ModelConfig(**model_config)
            parallel_config = ParallelConfig(**parallel_config)
            request_generator_config = RequestGeneratorConfig(
                **request_generator_config
            )
            request_config = RequestConfig(**request_config)
            server_config = ServerConfig(**server_config)

            # adding this to avoid port conflicts when running multiple jobs
            server_config.port = port

            if (
                not model_config.is_parallel_spec_valid(parallel_config.name)
                or not model_config.is_trace_valid(
                    request_generator_config.trace_request_interval_generator_trace_file
                )
                or (
                    model_config.name == "gpt-3.5-turbo"
                    and server_config.openai_server_engine
                    in ["vllm", "lightllm", "fastertransformers", "sarathi-serve"]
                )
                or (
                    request_generator_config.trace_file_name == "sharegpt"
                    and request_config.request_generator_max_tokens == 16384
                )
                or (
                    request_generator_config.trace_file_name == "arxiv"
                    and request_config.request_generator_max_tokens == 8192
                )
            ):
                continue

            port += 1

            job_config = cls(
                model_config,
                parallel_config,
                request_generator_config,
                request_config,
                server_config,
            )
            job_configs.append(job_config)

        return job_configs

    def __str__(self):
        return self.get_human_readable_name()


@dataclass
class BenchmarkConfig:
    output_dir: Optional[str] = None
    qps: Optional[float] = None
    should_use_given_dir: Optional[bool] = True
    ttft_deadline: Optional[float] = None
    tbt_deadline: Optional[float] = None
    wandb_group: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    should_write_metrics: Optional[bool] = True

    def to_config_dict(self):
        return {
            "output-dir": self.output_dir,
            "gamma-request-interval-generator-qps": self.qps,
            "poisson-request-interval-generator-qps": self.qps,
            "should-use-given-dir": self.should_use_given_dir,
            "ttft-deadline": self.ttft_deadline,
            "tbt-deadline": self.tbt_deadline,
            "wandb-group": self.wandb_group,
            "wandb-project": self.wandb_project,
            "wandb-run-name": self.wandb_run_name,
            "should-write-metrics": self.should_write_metrics,
        }

    def to_args(self):
        config_dict = self.to_config_dict()
        args = []
        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        args.append(f"--{key}")
                    else:
                        args.append(f"--no-{key}")
                else:
                    args.append(f"--{key} {value}")
        return " ".join(args)

    def get_run_id(self):
        return f"{self.qps}"

    def get_run_dir(self):
        return self.output_dir

    def to_human_readable_name(self):
        return f"QPS: {self.qps}, Run id: {self.get_run_id()}"
