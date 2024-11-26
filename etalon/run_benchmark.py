import argparse
import asyncio
import datetime
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import ray
from tqdm import tqdm

from etalon.core.hf_utils import get_tokenizer
from etalon.core.llm_clients import SUPPORTED_APIS
from etalon.core.request_config import RequestConfig
from etalon.core.requests_launcher import RequestsLauncher
from etalon.logger import init_logger
from etalon.metrics.service_metrics import ServiceMetrics
from etalon.request_generator.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)
from etalon.request_generator.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from etalon.request_generator.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)
from etalon.request_generator.length_generator.generator_registry import (
    RequestLengthGeneratorRegistry,
)
from etalon.request_generator.request_generator_config import RequestGeneratorConfig
from etalon.request_generator.utils import generate_random_prompt

logger = init_logger(__name__)


def get_request_params(
    model: str,
    llm_api: str,
    tokenizer: Any,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    request_length_generator: Optional[BaseRequestLengthGenerator] = None,
    corpus_lines: List[str] = None,
    address_append_value: Optional[str] = None,
    request_id: Optional[int] = None,
) -> Dict[str, Any]:
    (
        num_prompt_tokens,
        num_output_tokens,
    ) = request_length_generator.get_next_num_tokens()
    num_prompt_tokens = int(num_prompt_tokens)
    num_output_tokens = int(num_output_tokens)
    prompt = generate_random_prompt(
        tokenizer=tokenizer,
        num_prompt_tokens=num_prompt_tokens,
        num_output_tokens=num_output_tokens,
        corpus_lines=corpus_lines,
    )
    default_sampling_params = {"max_tokens": num_output_tokens}
    default_sampling_params.update(additional_sampling_params)
    request_config = RequestConfig(
        model=model,
        prompt=prompt,
        sampling_params=default_sampling_params,
        llm_api=llm_api,
        address_append_value=address_append_value,
        id=request_id,
    )

    return request_config


def should_send_new_request(
    service_metrics: ServiceMetrics, num_errored_requests_handled: int
) -> bool:
    """Check if a request should be sent based on the current state of the service.

    If the number of requests is less than the maximum number of requests, a request should always be sent.
    If the number of requests is greater than the maximum number of requests and not all errored requests are handled, a request should be sent.

    Args:
        service_metrics: The metrics for the service.
        num_errored_requests_handled: The number of errored requests handled.

    Returns:
        True if a request should be sent, False otherwise.
    """
    return (service_metrics.num_requests < service_metrics.max_requests) or (
        service_metrics.num_requests >= service_metrics.max_requests
        and num_errored_requests_handled < service_metrics.num_errored_requests
    )


async def collect_results(
    req_launcher: RequestsLauncher,
    service_metrics: ServiceMetrics,
    generated_texts: List[str],
) -> None:
    results = await req_launcher.collect_results()
    for out in results:
        request_metrics, generated_text = out
        if generated_text:
            service_metrics.add_request_metrics(request_metrics)
            generated_texts.append(generated_text)


async def run_main_loop(
    model: str,
    tokenizer_name: str,
    llm_api: str,
    tokenizer: Any,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    requests_interval_generator: Optional[BaseRequestIntervalGenerator] = None,
    requests_length_generator: Optional[BaseRequestLengthGenerator] = None,
    corpus_lines: List[str] = None,
    address_append_value: Optional[str] = None,
    request_every_minute: bool = False,
    service_metrics: ServiceMetrics = None,
    num_ray_clients: int = 2,
    num_concurrent_requests_per_client: int = 5,
    generated_texts: List[str] = None,
    pbar: tqdm = None,
):
    req_launcher = RequestsLauncher(
        model=model,
        tokenizer_name=tokenizer_name,
        llm_api=llm_api,
        num_ray_clients=num_ray_clients,
        num_concurrent_requests_per_client=num_concurrent_requests_per_client,
    )
    num_errored_requests_handled = 0
    await req_launcher.start()
    with service_metrics:
        while not service_metrics.should_stop():
            if should_send_new_request(service_metrics, num_errored_requests_handled):
                request_start_time = time.monotonic()
                if await req_launcher.is_free():
                    if service_metrics.num_requests >= service_metrics.max_requests:
                        num_errored_requests_handled += 1
                    service_metrics.register_launched_request()
                    request_config = get_request_params(
                        model=model,
                        llm_api=llm_api,
                        tokenizer=tokenizer,
                        additional_sampling_params=additional_sampling_params,
                        request_length_generator=requests_length_generator,
                        corpus_lines=corpus_lines.copy(),  # pass a copy of the corpus lines to avoid modifying the original
                        address_append_value=address_append_value,
                        request_id=service_metrics.num_requests,
                    )
                    await req_launcher.launch_requests(request_config)

                # poll less frequently when the number of requests is less than the max requests
                if not (service_metrics.num_requests % num_ray_clients):
                    await req_launcher.free_pool()
                    await collect_results(
                        req_launcher, service_metrics, generated_texts
                    )

                # sleep for the next request interval
                next_request_interval = (
                    60
                    if request_every_minute
                    else requests_interval_generator.get_next_inter_request_time()
                )
                while True:
                    if time.monotonic() - request_start_time >= next_request_interval:
                        break
            else:
                # just keep freeing pool and polling for results when no more requests can be sent.
                # If errored requests are encountered, they will be handled
                await req_launcher.free_pool()
                await collect_results(req_launcher, service_metrics, generated_texts)

            pbar.update(service_metrics.num_completed_requests - pbar.n)

    # wait for all requests to complete and collect all results
    await req_launcher.complete_tasks()
    await collect_results(req_launcher, service_metrics, generated_texts)
    # shut down clients and actors
    await req_launcher.shutdown()

    pbar.update(service_metrics.num_completed_requests - pbar.n)
    pbar.close()


def run_benchmark(
    model: str,
    tokenizer_name: str,
    output_dir: str,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_ray_clients: int = 2,
    num_concurrent_requests_per_client: int = 5,
    max_num_completed_requests: int = 500,
    timeout=90,
    llm_api: str = "openai",
    request_generator_config: RequestGeneratorConfig = None,
    ttft_deadline: float = 0.1,
    tbt_deadline: float = 0.05,
    target_deadline_miss_rate: float = 0.1,
    should_write_metrics: bool = True,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_run_name: str = None,
    address_append_value: Optional[str] = "chat/completions",
    request_every_minute: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_ray_clients: The number of ray actors to use for the benchmark. Each actor handles one LLM client.
        num_concurrent_requests_per_client: The number of concurrent requests per ray actor to make. Increase
            this to increase the amount of load and vice versa.
        timeout The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".
        request_interval_generator_provider: The name of the request generator provider to use for determining intervals.
        request_length_generator_provider: The name of the request generator provider to use for determining lengths.
        request_generator_config: The configuration for the request generator provider.
        ttft_deadline: The deadline for time to first token.
        tbt_deadline: The deadline between tokens.
        target_deadline_miss_rate: The target deadline miss rate.

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    service_metrics = ServiceMetrics(
        max_requests=max_num_completed_requests,
        timeout=timeout,
        ttft_deadline=ttft_deadline,
        tbt_deadline=tbt_deadline,
        target_deadline_miss_rate=target_deadline_miss_rate,
        should_write_metrics=should_write_metrics,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
    )

    tokenizer = get_tokenizer(
        tokenizer_name=tokenizer_name,
        trust_remote_code=True,
    )

    generated_texts = []

    pbar = tqdm(total=max_num_completed_requests)

    requests_interval_generator = RequestIntervalGeneratorRegistry.get_from_str(
        request_generator_config.request_interval_generator_provider,
        request_generator_config.get_request_interval_generator_config(),
    )
    requests_length_generator = RequestLengthGeneratorRegistry.get_from_str(
        request_generator_config.request_length_generator_provider,
        request_generator_config.get_request_length_generator_config(),
    )

    corpus_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    )
    with open(corpus_path, "r") as f:
        corpus_lines = f.readlines()

    asyncio.run(
        run_main_loop(
            model=model,
            tokenizer_name=tokenizer_name,
            llm_api=llm_api,
            tokenizer=tokenizer,
            additional_sampling_params=additional_sampling_params,
            requests_interval_generator=requests_interval_generator,
            requests_length_generator=requests_length_generator,
            corpus_lines=corpus_lines,
            address_append_value=address_append_value,
            request_every_minute=request_every_minute,
            service_metrics=service_metrics,
            num_ray_clients=num_ray_clients,
            num_concurrent_requests_per_client=num_concurrent_requests_per_client,
            generated_texts=generated_texts,
            pbar=pbar,
        )
    )

    logger.info(
        f"Results for token benchmark for {model} queried with the {llm_api} api. {service_metrics}"
    )

    service_metrics.store_output(output_dir)

    # store the generated texts
    with open(os.path.join(output_dir, "generated_texts.txt"), "w") as f:
        f.write(("\n" + "-" * 30 + "\n").join(generated_texts))


def parse_args():
    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument(
        "--model", type=str, required=True, help="The model to use for this load test."
    )
    args.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="The tokenizer to use for this load test. By default, the tokenizer is inferred from the model.",
    )
    args.add_argument(
        "--num-ray-clients",
        type=int,
        default=2,
        help=("The number of ray actors to use for benchmark. (default: %(default)s)"),
    )
    args.add_argument(
        "--num-concurrent-requests-per-client",
        type=int,
        default=5,
        help=(
            "The number of concurrent requests to send per ray actor (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="The amount of time to run the load test for. (default: %(default)s)",
    )
    args.add_argument(
        "--max-num-completed-requests",
        type=int,
        default=10,
        help=(
            "The number of requests to complete before finishing the test. Note "
            "that its possible for the test to timeout first. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--additional-sampling-params",
        type=str,
        default="{}",
        help=(
            "Additional sampling params to send with the each request to the LLM API. "
            "(default: %(default)s) No additional sampling params are sent."
        ),
    )
    args.add_argument(
        "--llm-api",
        type=str,
        default="openai",
        help=(
            f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
            " (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
    )
    args.add_argument(
        "--request-interval-generator-provider",
        type=str,
        default="gamma",
        help=(
            "The name of the request generator provider to use. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--request-length-generator-provider",
        type=str,
        default="zipf",
        help=("The name of the request length provider to use. (default: %(default)s)"),
    )
    args.add_argument(
        "--gamma-request-interval-generator-cv",
        type=float,
        default=0.5,
        help=(
            "The coefficient of variation for the gamma request interval generator. "
            "(default: %(default)s)"
        ),
    )
    args.add_argument(
        "--gamma-request-interval-generator-qps",
        type=float,
        default=0.2,
        help=(
            "The qps for the gamma request interval generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--poisson-request-interval-generator-qps",
        type=float,
        default=0.5,
        help=(
            "The qps for the poisson request interval generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--trace-request-interval-generator-trace-file",
        type=str,
        default=None,
        help=(
            "The trace file for the trace request interval generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--trace-request-interval-generator-start-time",
        type=str,
        default="1970-01-04 12:00:00",
        help=(
            "The start time for the trace request interval generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--trace-request-interval-generator-end-time",
        type=str,
        default="1970-01-04 15:00:00",
        help=(
            "The end time for the trace request interval generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--trace-request-interval-generator-time-scale-factor",
        type=float,
        default=0.3,
        help=(
            "The time scale factor for the trace request interval generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--fixed-request-generator-prefill-tokens",
        type=int,
        default=2048,
        help=(
            "The number of tokens to prefill the fixed request generator with. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--fixed-request-generator-decode-tokens",
        type=int,
        default=256,
        help=(
            "The number of tokens to decode the fixed request generator with. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--synthetic-request-generator-min-tokens",
        type=int,
        default=1024,
        help=(
            "The minimum number of tokens to generate for the synthetic request generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--request-generator-max-tokens",
        type=int,
        default=40000,
        help=("The maximum number of tokens to generate. (default: %(default)s)"),
    )
    args.add_argument(
        "--synthetic-request-generator-prefill-to-decode-ratio",
        type=float,
        default=10,
        help=(
            "The prefill to decode ratio for the synthetic request generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--zipf-request-length-generator-theta",
        type=float,
        default=0.4,
        help=(
            "The theta value for the zipf request length generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--zipf-request-length-generator-scramble",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Whether to scramble the zipf request length generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--trace-request-length-generator-trace-file",
        type=str,
        default=None,
        help=(
            "The trace file for the trace request length generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--trace-request-length-generator-prefill-scale-factor",
        type=float,
        default=1,
        help=(
            "The prefill scale factor for the trace request length generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--trace-request-length-generator-decode-scale-factor",
        type=float,
        default=1,
        help=(
            "The decode scale factor for the trace request length generator. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--seed",
        type=int,
        default=42,
        help=("The seed for the request generator. (default: %(default)s)"),
    )
    args.add_argument(
        "--ttft-deadline",
        type=float,
        default=0.1,
        help=("The deadline for time to first token. (default: %(default)s)"),
    )
    args.add_argument(
        "--tbt-deadline",
        type=float,
        default=0.05,
        help=("The deadline between tokens. (default: %(default)s)"),
    )
    args.add_argument(
        "--target-deadline-miss-rate",
        type=float,
        default=0.1,
        help=("The target miss rate. (default: %(default)s)"),
    )
    args.add_argument(
        "--should-use-given-dir",  # added to prevent the creation of a new directories for the capacity search
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Whether to add directly use --output-dir directory or create new directories for the results. (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--should-write-metrics",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=("Whether to write metrics to wandb. (default: %(default)s)"),
    )
    args.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help=("The wandb project name. (default: %(default)s)"),
    )
    args.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help=("The wandb entity name. (default: %(default)s)"),
    )
    args.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help=("The wandb group name. (default: %(default)s)"),
    )
    args.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help=("The wandb run name. (default: %(default)s)"),
    )
    args.add_argument(
        "--address-append-value",
        type=str,
        default="chat/completions",
        help=("The address append value for OpenAI API. (default: %(default)s)"),
    )
    args.add_argument(
        "--request-every-minute",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=("Whether to request every minute. (default: %(default)s)"),
    )
    args.add_argument(
        "--time-stamp",
        type=str,
        default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        help=("The time stamp for the benchmark. (default: %(default)s)"),
    )
    args.add_argument(
        "--prefill-lengths",
        type=int,
        nargs="+",
        default=[],
        help=(
            "The list of prefill lengths for the prefill profiler. (default: %(default)s)"
        ),
    )

    args = args.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model

    if not args.should_use_given_dir:
        benchmark_identifier = f"{args.model}_{args.request_interval_generator_provider}_{args.request_length_generator_provider}"
        benchmark_identifier = re.sub(r"[^\w\d-]+", "-", benchmark_identifier)
        benchmark_identifier = re.sub(r"-{2,}", "-", benchmark_identifier)

        # create a directory to store the results with date and time
        args.output_dir = os.path.join(
            args.output_dir,
            benchmark_identifier,
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )

    if args.additional_sampling_params:
        args.additional_sampling_params = json.loads(args.additional_sampling_params)
    else:
        args.additional_sampling_params = {}

    # dump config to a file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return args


if __name__ == "__main__":
    random.seed(11111)

    ray.init(runtime_env={"env_vars": dict(os.environ)})

    args = parse_args()

    request_generator_config = RequestGeneratorConfig(args=args)

    run_benchmark(
        llm_api=args.llm_api,
        output_dir=args.output_dir,
        model=args.model,
        tokenizer_name=args.tokenizer,
        timeout=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        num_ray_clients=args.num_ray_clients,
        num_concurrent_requests_per_client=args.num_concurrent_requests_per_client,
        additional_sampling_params=args.additional_sampling_params,
        request_generator_config=request_generator_config,
        ttft_deadline=args.ttft_deadline,
        tbt_deadline=args.tbt_deadline,
        target_deadline_miss_rate=args.target_deadline_miss_rate,
        should_write_metrics=args.should_write_metrics,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        address_append_value=args.address_append_value,
        request_every_minute=args.request_every_minute,
    )
