from typing import Any, List

from metron.core.llm_clients.base_llm_client import BaseLLMClient
from metron.core.request_config import RequestConfig

import asyncio


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(self, llm_clients: List[BaseLLMClient]):
        self.llm_client_free_pool = asyncio.Queue(maxsize=len(llm_clients))
        self.llm_client_busy_pool = asyncio.Queue(maxsize=len(llm_clients))
        self.results_queue = asyncio.Queue()
        self.llm_clients = llm_clients
        for client_id in range(len(self.llm_clients)):
            self.llm_client_free_pool.put_nowait(client_id)

    async def start_tasks(self):
        self.client_tasks = [asyncio.create_task(self.send_requests()) for _ in self.llm_clients]

    async def send_requests(self) -> None:
        while True:
            task = await self.llm_client_busy_pool.get()
            if task is None:
                self.llm_client_busy_pool.task_done()
                break
            client_id: int = task[0]
            request_config: RequestConfig = task[1]
            request_metrics, generated_text = await self.llm_clients[client_id].send_llm_request(request_config)
            await self.results_queue.put((request_metrics, generated_text))
            await self.llm_client_free_pool.put(client_id)
            self.llm_client_busy_pool.task_done()

    async def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        free_client_id = await self.llm_client_free_pool.get()
        await self.llm_client_busy_pool.put((free_client_id, request_config))

    def get_results(self) -> List[Any]:
        """Return results that are ready from completed requests.

        Returns:
            A list of results that are ready.

        """
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get_nowait())
        return results

    async def complete(self):
        for _ in range(len(self.llm_clients)):
            await self.llm_client_busy_pool.put(None)
        for task in self.client_tasks:
            task.cancel()
