import abc
from typing import Tuple

from etalon.core.hf_utils import get_tokenizer
from etalon.core.request_config import RequestConfig
from etalon.metrics.request_metrics import RequestMetrics


class BaseLLMClient:
    """A client for making requests to a LLM API e.g Anyscale Endpoints."""

    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = get_tokenizer(
            tokenizer_name,
            trust_remote_code=True,
        )

    def get_token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @abc.abstractmethod
    async def send_llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[RequestMetrics, str]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
        """
        ...
