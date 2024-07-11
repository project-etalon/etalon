import asyncio
from typing import Any, List

import ray

from metron.core.llm_clients.base_llm_client import BaseLLMClient
from metron.core.request_config import RequestConfig


@ray.remote
class AsyncRequestsManager:
    """Manages requests for single LLM API client."""

    def __init__(
        self, client_id: int, llm_client: BaseLLMClient, max_concurrent_requests: int
    ):
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
        self.client_tasks = [
            asyncio.create_task(self.process_requests(i))
            for i in range(self.max_concurrent_requests)
        ]

    async def process_requests(self, i) -> None:
        while True:
            request_config = await self.requests_queue.get()
            if request_config is None:
                break
            await self.handle_request(request_config, i)
            self.requests_queue.task_done()

    async def handle_request(self, request_config: RequestConfig, i: int) -> None:
        """Handle requests to the LLM API.

        Returns:
            None
        """
        if request_config:
            result = await self.llm_client.send_llm_request(request_config)
            self.results.append(result)

    async def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        await self.requests_queue.put(request_config)

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
        for task in self.client_tasks:
            await task
