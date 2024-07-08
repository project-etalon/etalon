from typing import List

from metron.core.llm_clients.base_llm_client import BaseLLMClient
from metron.core.llm_clients.litellm_client import LiteLLMClient
from metron.core.llm_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from metron.core.llm_clients.sagemaker_client import SageMakerClient
from metron.core.llm_clients.vertexai_client import VertexAIClient

SUPPORTED_APIS = ["openai", "anthropic", "litellm"]


def construct_clients(
    model_name: str, llm_api: str, num_clients: int
) -> List[BaseLLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        clients = [
            OpenAIChatCompletionsClient.remote(model_name) for _ in range(num_clients)
        ]
    elif llm_api == "sagemaker":
        clients = [SageMakerClient.remote(model_name) for _ in range(num_clients)]
    elif llm_api == "vertexai":
        clients = [VertexAIClient.remote(model_name) for _ in range(num_clients)]
    elif llm_api in SUPPORTED_APIS:
        clients = [LiteLLMClient.remote(model_name) for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients
