from typing import Any, List

from ray.util import ActorPool

from metron.core.llm_clients.base_llm_client import BaseLLMClient
from metron.core.request_config import RequestConfig


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(self, llm_clients: List[BaseLLMClient]):
        self.llm_client_pool = ActorPool(llm_clients)

    def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        if self.llm_client_pool.has_free():
            self.llm_client_pool.submit(
                lambda client, _request_config: client.send_llm_request.remote(
                    _request_config
                ),
                request_config,
            )

    def get_next_ready(self, block: bool = False) -> List[Any]:
        """Return results that are ready from completed requests.

        Args:
            block: Whether to block until a result is ready.

        Returns:
            A list of results that are ready.

        """
        results = []
        if not block:
            while self.llm_client_pool.has_next():
                results.append(self.llm_client_pool.get_next_unordered())
        else:
            while not self.llm_client_pool.has_next():
                pass
            while self.llm_client_pool.has_next():
                results.append(self.llm_client_pool.get_next_unordered())
        return results
