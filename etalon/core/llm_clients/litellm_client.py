import time
from typing import Tuple

import ray

from etalon.core.llm_clients.base_llm_client import BaseLLMClient
from etalon.core.request_config import RequestConfig
from etalon.logger import init_logger
from etalon.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


class LiteLLMClient(BaseLLMClient):
    """Client for LiteLLM Completions API."""

    async def send_llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[RequestMetrics, str]:
        # litellm package isn't serializable, so we import it within the function
        # to maintain compatibility with ray.
        from litellm import completion, validate_environment

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        assert (
            request_config.llm_api is not None
        ), "the request config's llm_api must be set."
        if request_config.llm_api == "litellm":
            model = request_config.model
        else:
            model = request_config.llm_api + "/" + request_config.model
        validation_result = validate_environment(model)
        if validation_result["missing_keys"]:
            raise ValueError(
                f"The following environment vars weren't found but were necessary for "
                f"the model {request_config.model}: {validation_result['missing_keys']}"
            )
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        inter_token_times = []
        tokens_received = 0
        error_msg = None
        error_response_code = None
        generated_text = ""

        most_recent_received_token_time = time.monotonic()

        try:
            response = completion(**body)
            for tok in response:
                if tok.choices[0].delta:
                    delta = tok.choices[0].delta
                    if delta.get("content", None):
                        inter_token_times.append(
                            time.monotonic() - most_recent_received_token_time
                        )
                        generated_text += delta["content"]
                        most_recent_received_token_time = time.monotonic()
                        tokens_received += 1
        except Exception as e:
            logger.error(f"Warning Or Error: ({error_response_code}) {e}")
            error_msg = str(e)

        metrics = RequestMetrics(
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
        )

        return metrics, generated_text
