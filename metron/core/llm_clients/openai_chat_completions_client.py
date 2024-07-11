import json
import os
import time
from typing import List, Tuple

import httpx
from metron.core.llm_clients.base_llm_client import BaseLLMClient
from metron.core.request_config import RequestConfig
from metron.logger import init_logger
from metron.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)

# Maximum number of responses to store for token counting
MAX_RESPONSES_ALLOWED_TO_STORE = 5


class OpenAIChatCompletionsClient(BaseLLMClient):
    """Client for OpenAI Chat Completions API."""

    def total_tokens(self, response_list: List[str]) -> int:
        merged_content = "".join(response_list)
        return self.get_token_length(merged_content)

    def get_current_tokens_received(
        self,
        previous_responses: List[str],
        current_response: str,
        previous_token_count: int,
    ) -> Tuple[int, int]:
        previous_responses.append(current_response)
        current_tokens_received = (
            self.total_tokens(previous_responses) - previous_token_count
        )
        if len(previous_responses) > MAX_RESPONSES_ALLOWED_TO_STORE:
            previous_responses.pop(0)
        previous_token_count = self.total_tokens(previous_responses)
        return current_tokens_received, previous_token_count

    async def send_llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[RequestMetrics, str]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        address = os.environ.get("OPENAI_API_BASE")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")
        headers = {"Authorization": f"Bearer {key}"}
        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += request_config.address_append_value or "chat/completions"

        inter_token_times = []
        error_msg = None
        error_response_code = None
        tokens_received = 0
        generated_text = ""
        previous_responses = []
        previous_token_count = 0

        most_recent_received_token_time = time.monotonic()

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", 
                    address, 
                    json=body, 
                    timeout=180, 
                    headers=headers
                ) as response:
                    if response.status_code != 200:
                        error_msg = response.text
                        error_response_code = response.status_code
                        logger.error(f"Request Error: {response.content}")
                        response.raise_for_status()

                    async for chunk in response.aiter_lines():
                        chunk = chunk.strip()

                        if not chunk:
                            continue
                        stem = "data: "
                        chunk = chunk[len(stem) :]
                        if chunk == b"[DONE]":
                            continue
                        elif chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)

                        if "error" in data:
                            error_msg = data["error"]["message"]
                            error_response_code = data["error"]["code"]
                            raise RuntimeError(data["error"]["message"])

                        delta = data["choices"][0]["delta"]
                        if delta.get("content", None):
                            (
                                current_tokens_received,
                                previous_token_count,
                            ) = self.get_current_tokens_received(
                                previous_responses=previous_responses,
                                current_response=delta["content"],
                                previous_token_count=previous_token_count,
                            )

                            tokens_received += current_tokens_received
                            inter_token_times.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                            if current_tokens_received > 1:
                                inter_token_times.extend(
                                    [0] * (current_tokens_received - 1)
                                )
                            most_recent_received_token_time = time.monotonic()
                            generated_text += delta["content"]
        except Exception as e:
            logger.error(f"{request_config.id,} Warning Or Error: ({error_response_code}) {e}")

        metrics = RequestMetrics(
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
        )

        return metrics, generated_text
