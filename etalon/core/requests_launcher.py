import asyncio
from typing import Any, List

from ray.util import ActorPool

from etalon.core.request_config import RequestConfig
from etalon.core.requests_manager import AsyncRequestsManager


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        model: str,
        tokenizer_name: str,
        llm_api: str,
        num_ray_clients: int,
        num_concurrent_requests_per_client: int,
    ):
        self.actors = []
        for client_id in range(num_ray_clients):
            self.actors.append(
                AsyncRequestsManager.remote(
                    client_id=client_id,
                    model=model,
                    tokenizer_name=tokenizer_name,
                    llm_api=llm_api,
                    max_concurrent_requests=num_concurrent_requests_per_client,
                )
            )
        self.llm_client_pool = ActorPool(self.actors)

    async def start(self) -> None:
        """Starts the tasks on each actor to handle requests.

        Returns:
            None

        """
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

    async def is_free(self) -> bool:
        """Check if the pool of actors is free.

        Returns:
            True if the pool of actors is free, False otherwise.

        """
        return self.llm_client_pool.has_free()

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
            while len(self.llm_client_pool._pending_submits) > 0:
                await asyncio.sleep(0.1)
                pass
            while self.llm_client_pool.has_next():
                self.llm_client_pool.get_next_unordered()

    async def complete_tasks(self) -> None:
        """Complete all tasks"""
        await self.free_pool(block=True)
        for actor in self.actors:
            await actor.complete_tasks.remote()

    async def collect_results(self) -> List[Any]:
        """Collect results from the actors.

        Returns:
            A list of results from the actors.

        """
        results = []
        for actor in self.actors:
            results.extend(await actor.get_results.remote())
        return results

    async def shutdown(self) -> None:
        """Shutdown the pool of actors.

        Returns:
            None

        """
        for actor in self.actors:
            await actor.shutdown.remote()
