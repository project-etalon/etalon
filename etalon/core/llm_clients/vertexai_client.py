import json
import os
import time
from typing import Tuple

import ray
import requests

from etalon.core.llm_clients.base_llm_client import BaseLLMClient
from etalon.core.request_config import RequestConfig
from etalon.logger import init_logger
from etalon.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


class VertexAIClient(BaseLLMClient):
    """Client for VertexAI API."""

    async def send_llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[RequestMetrics, str]:
        project_id = os.environ.get("GCLOUD_PROJECT_ID")
        region = os.environ.get("GCLOUD_REGION")
        endpoint_id = os.environ.get("VERTEXAI_ENDPOINT_ID")
        access_token = os.environ.get("GCLOUD_ACCESS_TOKEN").strip()
        if not project_id:
            raise ValueError("the environment variable GCLOUD_PROJECT_ID must be set.")
        if not region:
            raise ValueError("the environment variable GCLOUD_REGION must be set.")
        if not endpoint_id:
            raise ValueError(
                "the environment variable VERTEXAI_ENDPOINT_ID must be set."
            )
        if not access_token:
            raise ValueError(
                "the environment variable GCLOUD_ACCESS_TOKEN must be set."
            )
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        inter_token_times = []
        tokens_received = 0
        error_msg = None
        error_response_code = None
        generated_text = ""

        try:
            # Define the URL for the request
            url = (
                f"https://{region}-aiplatform.googleapis.com/v1/projects/"
                f"{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"
            )

            # Define the headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            sampling_params = request_config.sampling_params
            if "max_new_tokens" in sampling_params:
                sampling_params["maxOutputTokens"] = sampling_params.pop(
                    "max_new_tokens"
                )

            # Define the data payload
            data = {"instances": [{"prompt": prompt}], "parameters": sampling_params}

            # Make the POST request
            start_time = time.monotonic()
            response = requests.post(url, headers=headers, data=json.dumps(data))
            total_request_time = time.monotonic() - start_time
            response_code = response.status_code
            response.raise_for_status()
            # output from the endpoint is in the form:
            # {"predictions": ["Input: ... \nOutput:\n ..."]}
            generated_text = response.json()["predictions"][0].split("\nOutput:\n")[1]
            tokens_received = self.get_token_length(generated_text)
            inter_token_times = [
                total_request_time / tokens_received for _ in range(tokens_received)
            ]
        except Exception as e:
            logger.error(f"Warning Or Error: ({error_response_code}) {e}")
            error_msg = str(e)
            error_response_code = response_code

        metrics = RequestMetrics(
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
        )

        return metrics, generated_text


if __name__ == "__main__":
    # Run these before hand:

    # gcloud auth application-default login
    # gcloud config set project YOUR_PROJECT_ID
    # export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
    # export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
    # export GCLOUD_REGION=YOUR_REGION
    # export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

    client = VertexAIClient.remote()
    request_config = RequestConfig(
        prompt=("Give me ten interview questions for the role of program manager.", 10),
        model="gpt3",
        sampling_params={
            "temperature": 0.2,
            "max_new_tokens": 256,
            "top_k": 40,
            "top_p": 0.95,
        },
    )
    ray.get(client.send_llm_request.remote(request_config))
