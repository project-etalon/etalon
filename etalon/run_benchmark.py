import argparse
import datetime
import json
import os
import random
import re
import time
import threading
from queue import Empty
from multiprocessing import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from etalon.config import BenchmarkConfig
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
    """Check if a request should be sent based on the current state of the service."""
    return (service_metrics.num_requests < service_metrics.max_requests) or (
        service_metrics.num_requests >= service_metrics.max_requests
        and num_errored_requests_handled < service_metrics.num_errored_requests
    )


def dispatch_requests(
    input_queue: Queue,
    service_metrics: ServiceMetrics,
    model: str,
    llm_api: str,
    tokenizer: Any,
    additional_sampling_params: Dict[str, Any],
    requests_interval_generator: BaseRequestIntervalGenerator,
    requests_length_generator: BaseRequestLengthGenerator,
    corpus_lines: List[str],
    address_append_value: str,
    stop_event: threading.Event,
) -> None:
    """Thread function to generate and dispatch requests."""
    num_errored_requests_handled = 0

    while not stop_event.is_set():
        if should_send_new_request(service_metrics, num_errored_requests_handled):
            request_start_time = time.monotonic()

            # Check if we should handle error request
            if service_metrics.num_requests >= service_metrics.max_requests:
                num_errored_requests_handled += 1
            
            # Create and dispatch request
            service_metrics.register_launched_request()
            request_config = get_request_params(
                model=model,
                llm_api=llm_api,
                tokenizer=tokenizer,
                additional_sampling_params=additional_sampling_params,
                request_length_generator=requests_length_generator,
                corpus_lines=corpus_lines.copy(),
                address_append_value=address_append_value,
                request_id=service_metrics.num_requests,
            )
            input_queue.put(request_config)

            # Wait for next interval
            next_request_interval = requests_interval_generator.get_next_inter_request_time()
            while not stop_event.is_set():
                if time.monotonic() - request_start_time >= next_request_interval:
                    break
                time.sleep(0.01)
        else:
            time.sleep(0.01)


def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_texts: List[str],
    pbar: tqdm,
    stop_event: threading.Event,
) -> None:
    """Thread function to process results from the output queue."""
    while not stop_event.is_set() or not output_queue.empty():
        try:
            result = output_queue.get(timeout=0.1)
            request_metrics, generated_text = result
            if generated_text:
                service_metrics.add_request_metrics(request_metrics)
                generated_texts.append(generated_text)
            
            pbar.update(service_metrics.num_completed_requests - pbar.n)
        except Empty:
            continue


def run_main_loop(
    benchmark_config: BenchmarkConfig,
    generated_texts: List[str] = None,
    pbar: tqdm = None,
):
    """Run the main loop for the benchmark."""

    logger.info("Starting the main loop.")

    # Create queues for commmunication
    input_queue = Queue()
    output_queue = Queue()
    stop_event = threading.Event()

    # Initialize request launcher
    req_launcher = RequestsLauncher(
        model=model,
        tokenizer_name=tokenizer_name,
        llm_api=llm_api,
        num_clients=num_clients,
        num_concurrent_requests_per_client=num_concurrent_requests_per_client,
        input_queue=input_queue,
        output_queue=output_queue,
    )

    # Start the request launcher processes
    req_launcher.start()

    # Create and start producer-consumer threads
    dispatcher_thread = Thread(
        target=dispatch_requests,
        args=(
            input_queue,
            service_metrics,
            model,
            llm_api,
            tokenizer,
            additional_sampling_params,
            requests_interval_generator,
            requests_length_generator,
            corpus_lines,
            address_append_value,
            stop_event,
        ),
    )

    processor_thread = Thread(
        target=process_results,
        args=(
            output_queue,
            service_metrics,
            generated_texts,
            pbar,
            stop_event,
        ),
    )

    dispatcher_thread.start()
    processor_thread.start()

    # Monitor and wait for completion
    with service_metrics:
        while not service_metrics.should_stop():
            time.sleep(0.1)
        logger.info("Stopping the main loop.")

    # Signal threads to stop and wait for completion
    stop_event.set()
    dispatcher_thread.join()
    processor_thread.join()

    # Terminate all clients
    req_launcher.complete_tasks()

    pbar.close()
    logger.info("Main loop completed.")


def run_benchmark(
    benchmark_config: BenchmarkConfig,
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

    run_main_loop(
        benchmark_config=benchmark_config,
        generated_texts=generated_texts,
        pbar=pbar,
    )

    logger.info(
        f"Results for token benchmark for {model} queried with the {llm_api} api. {service_metrics}"
    )

    service_metrics.store_output(output_dir)
    logger.info(f"Metrics stored to {output_dir}")

    # store the generated texts
    with open(os.path.join(output_dir, "generated_texts.txt"), "w") as f:
        f.write(("\n" + "-" * 30 + "\n").join(generated_texts))
    
    os._exit(0)


if __name__ == "__main__":
    config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    random.seed(config.seed)

    # TODO: update
    request_generator_config = RequestGeneratorConfig(args=None)

    # TODO: update
    run_benchmark(
        llm_api=args.llm_api,
        output_dir=args.output_dir,
        model=args.model,
        tokenizer_name=args.tokenizer,
        timeout=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        num_clients=args.num_clients,
        num_concurrent_requests_per_client=args.num_concurrent_requests_per_client,
        additional_sampling_params=args.additional_sampling_params,
        request_generator_config=request_generator_config,
        ttft_deadline=args.ttft_deadline,
        tbt_deadline=args.tbt_deadline,
        target_deadline_miss_rate=args.target_deadline_miss_rate,
        should_write_metrics=args.should_write_metrics,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        address_append_value=args.address_append_value,
    )
