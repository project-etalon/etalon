from typing import Any, List

from metron.core.llm_clients.base_llm_client import BaseLLMClient
from metron.core.request_config import RequestConfig

import asyncio
import ray
from ray.actor import exit_actor


@ray.remote
class AsyncRequestsManager:
    """Manages requests for single LLM API client."""

    def __init__(self, client_id: int, llm_client: BaseLLMClient, max_concurrent_requests: int):
        self.max_concurrent_requests = max_concurrent_requests
        self.requests_queue = asyncio.Queue(maxsize=max_concurrent_requests)
        self.results = []
        self.llm_client = llm_client
        self.client_id = client_id

    async def start_tasks(self):
        """Starts the tasks to handle requests.

        Returns:
            None
        """
        self.client_tasks = [asyncio.create_task(self.process_requests(i)) for i in range(self.max_concurrent_requests)]

    async def process_requests(self, i) -> None:
        while True:
            print(f"{self.client_id, i} waiting for request", flush=True)
            request_config = await self.requests_queue.get()
            print(f"{self.client_id, i} got request {request_config.id if request_config is not None else None}", flush=True)
            if request_config is None:
                print(f"{self.client_id, i} Exiting", flush=True)
                break
            print(f"{self.client_id, i} handling request {request_config.id}", flush=True)
            await self.handle_request(request_config, i)
            print(f"{self.client_id, i} completed request {request_config.id}", flush=True)
            self.requests_queue.task_done()

    async def handle_request(self, request_config: RequestConfig, i: int) -> None:
        """Handle requests to the LLM API.

        Returns:
            None
        """
        if request_config:
            print(f"{self.client_id, i} sending request {request_config.id}", flush=True)
            # await asyncio.sleep(10)
            result = await self.llm_client.send_llm_request(request_config)
            await asyncio.sleep(60)
            print(f"{self.client_id, i} got result for request {request_config.id}", flush=True)
            self.results.append(result)

    async def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        print(f"{self.client_id} Queuing request {request_config.id}", flush=True)
        await self.requests_queue.put(request_config)
        print(f"{self.client_id} Queued request {request_config.id}", flush=True)

    async def get_results(self) -> List[Any]:
        """Return results that are ready from completed requests.

        Returns:
            A list of results that are ready.

        """
        return self.results

    async def complete_tasks(self):
        """Waits for all tasks to complete.

        Returns:
            None
        """
        for _ in range(self.max_concurrent_requests):
            await self.requests_queue.put(None)
        print(f"{self.client_id} Waiting for tasks to complete")
        for task in self.client_tasks:
            await task
        print(f"{self.client_id} Tasks completed")
