from typing import List

from .base_llm_client import BaseLLMClient
from .litellm_client import LiteLLMClient
from .openai_chat_completions_client import OpenAIChatCompletionsClient
from .sagemaker_client import SageMakerClient
from .vertexai_client import VertexAIClient

SUPPORTED_APIS = ["openai", "anthropic", "litellm"]


def construct_clients(
    model_name: str,
    tokenizer_name: str,
    llm_api: str,
    num_clients: int,
    use_ray: bool = True,
) -> List[BaseLLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        impl = OpenAIChatCompletionsClient
    elif llm_api == "sagemaker":
        impl = SageMakerClient
    elif llm_api == "vertexai":
        impl = VertexAIClient
    elif llm_api in SUPPORTED_APIS:
        impl = LiteLLMClient
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    if use_ray:
        clients = [impl.remote(model_name, tokenizer_name) for _ in range(num_clients)]
    else:
        clients = [impl(model_name, tokenizer_name) for _ in range(num_clients)]

    return clients
