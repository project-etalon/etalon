import math
from typing import Any, List

import asyncio
from ray.util import ActorPool

from metron.core.llm_clients import construct_clients
from metron.core.request_config import RequestConfig
from metron.core.requests_manager import AsyncRequestsManager

class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(self, model: str, llm_api: str, num_concurrent_requests: int):
        # ray clients = sqrt(num_concurrent_requests, so that each client can handle sqrt(num_concurrent_requests)
        num_ray_actors = int(math.floor(math.sqrt(num_concurrent_requests)))
        llm_clients = construct_clients(
            model_name=model,
            llm_api=llm_api,
            num_clients=num_ray_actors,
            use_ray=False
        )
        self.actors = []
        for client_id, client in enumerate(llm_clients):
            num_concurrent_requests_per_actor = num_concurrent_requests // num_ray_actors
            if client_id == 0:
                # when num_concurrent_requests is not perfect square
                num_concurrent_requests_per_actor += num_concurrent_requests % num_ray_actors
            self.actors.append(
                AsyncRequestsManager.remote(client_id, client, max_concurrent_requests=num_concurrent_requests_per_actor)
            )
        self.llm_client_pool = ActorPool(self.actors)
        
    async def start(self):
        for actor in self.actors:
            await actor.start_tasks.remote()

    async def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        self.llm_client_pool.submit(
            lambda actor, _request_config: actor.launch_requests.remote(
                _request_config
            ),
            request_config,
        )

    async def free_pool(self, block: bool = False) -> None:
        """Frees the pool of actors for the next batch of requests.

        Args:
            block: Whether to block until a result is ready.

        Returns:
            None

        """
        if not block:
            while self.llm_client_pool.has_next():
                self.llm_client_pool.get_next_unordered()
        else:
            while not self.llm_client_pool.has_next():
                pass
            while self.llm_client_pool.has_next():
                self.llm_client_pool.get_next_unordered()

    async def complete(self):
        """Complete all tasks"""
        while self.llm_client_pool.has_next():
            self.llm_client_pool.get_next_unordered()
        for actor in self.actors:
            await actor.complete.remote()

    async def get_results(self) -> List[Any]:
        """Return results that are ready from completed requests.

        Returns:
            A list of results that are ready.

        """
        results = []
        for actor in self.actors:
            results.extend(await actor.get_results.remote())
        return results
