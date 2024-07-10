from typing import Any, List

from metron.core.llm_clients.base_llm_client import BaseLLMClient
from metron.core.request_config import RequestConfig

import asyncio
import ray


@ray.remote
class AsyncRequestsManager:
    """Manages requests for single LLM API client."""

    def __init__(self, client_id: int, llm_client: BaseLLMClient, max_concurrent_requests: int):
        self.max_concurrent_requests = max_concurrent_requests
        self.busy_pool = asyncio.Queue(maxsize=max_concurrent_requests)
        self.results_queue = asyncio.Queue()
        self.llm_client = llm_client
        self.client_id = client_id

    async def start_tasks(self):
        self.client_tasks = [asyncio.create_task(self.send_requests()) for _ in range(self.max_concurrent_requests)]

    async def send_requests(self) -> None:
        while True:
            request_config = await self.busy_pool.get()
            if request_config is None:
                self.busy_pool.task_done()
                break
            request_metrics, generated_text = await self.llm_client.send_llm_request(request_config)
            await self.results_queue.put((request_metrics, generated_text))
            self.busy_pool.task_done()

    async def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        await self.busy_pool.put(request_config)

    async def get_results(self) -> List[Any]:
        """Return results that are ready from completed requests.

        Returns:
            A list of results that are ready.

        """
        results = []
        # Get all results from the queue
        while not self.results_queue.empty():
            result = await self.results_queue.get()
            results.append(result)
            self.results_queue.task_done()
        return results

    async def complete(self):
        """Waits for all tasks to complete.

        Returns:
            None
        """
        for _ in range(self.max_concurrent_requests):
            await self.busy_pool.put(None)
        for task in self.client_tasks:
            await task
