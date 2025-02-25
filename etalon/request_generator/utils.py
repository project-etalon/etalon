import math
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from etalon.config.config import ClientConfig
from etalon.core.request_config import RequestConfig
from etalon.logger import init_logger
from etalon.request_generator.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)
from etalon.request_generator.length_generator.trace_generator import (
    PrefixRequestLengthGenerator,
)
from etalon.request_generator.utils import generate_random_prompt

logger = init_logger(__name__)


def generate_random_prompt(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    num_prompt_tokens: int = 1024,
    num_output_tokens: int = 128,
    corpus_lines: Union[List[str], None] = None,
    add_instruction: bool = True,
) -> Tuple[str, int]:
    """Generate a random prompt with a given number of tokens.

    Args:
        num_prompt_tokens: The number of tokens to generate in the prompt.
        num_output_tokens: The number of tokens to expect in the output.

        The prompt will be generated such that the output
        will be approximately this many tokens.

    Returns:
        A random prompt with the given number of tokens.
    """
    assert corpus_lines is not None, "corpus_lines must be provided"

    get_token_length = lambda text: len(tokenizer.encode(text))

    instruction = (
        'INSTRUCTION: Mimic below text enclosed in """ quotes and generate '
        f"long text of at least {num_output_tokens} tokens.\n\n"
    )

    remaining_prompt_tokens = num_prompt_tokens - get_token_length(instruction)
    random.shuffle(corpus_lines)
    sampling_lines = True
    prompt = (instruction + '"""') if add_instruction else ""
    remaining_prompt_tokens -= get_token_length(prompt) * 2
    while sampling_lines:
        for line in corpus_lines:
            line_to_add = line
            if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                # This will cut off a line in the middle of a word, but that's ok since an
                # llm should be able to handle that.
                line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                sampling_lines = False
                prompt += line_to_add
                break
            prompt += line_to_add
            remaining_prompt_tokens -= get_token_length(line_to_add)

    if add_instruction:
        prompt += '"""'
    return (prompt, num_prompt_tokens)


class BaseRequestGenerator(ABC):

    def __init__(
        self,
        client_config: ClientConfig,
        tokenizer: Any,
        request_length_generator: BaseRequestLengthGenerator,
        corpus_lines: Optional[List[str]] = None,
    ):
        self.client_config = client_config
        self.tokenizer = tokenizer
        self.request_length_generator = request_length_generator
        self.corpus_lines = corpus_lines

    @abstractmethod
    def get_request_params(
        self,
        request_id: Optional[int] = None,
    ) -> RequestConfig:
        raise NotImplementedError("Subclasses must implement generate_request method")


class RequestGenerator(BaseRequestGenerator):

    def get_request_params(
        self,
        request_id: Optional[int] = None,
    ) -> RequestConfig:

        (
            num_prompt_tokens,
            num_output_tokens,
        ) = self.request_length_generator.get_next_num_tokens()
        if num_prompt_tokens < 0 or num_output_tokens < 0:
            logger.error(
                f"Invalid number of tokens generated: prompt={num_prompt_tokens}, output={num_output_tokens} (potentially from trace request length generator)."
            )
        num_prompt_tokens = int(num_prompt_tokens)
        num_output_tokens = int(num_output_tokens)
        prompt = generate_random_prompt(
            tokenizer=self.tokenizer,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            corpus_lines=self.corpus_lines,
        )
        default_sampling_params = {"max_tokens": num_output_tokens}
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )
        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=prompt,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=request_id,
        )

        return request_config


class PrefixBasedRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        client_config: ClientConfig,
        tokenizer: Any,
        request_length_generator: BaseRequestLengthGenerator,
        corpus_lines: Optional[List[str]] = None,
    ):
        super().__init__(
            client_config=client_config,
            tokenizer=tokenizer,
            request_length_generator=request_length_generator,
            corpus_lines=corpus_lines,
        )
        self.past_prompts: dict[str, str] = {}

    def get_request_params(
        self,
        request_id: Optional[int] = None,
    ) -> RequestConfig:
        assert isinstance(self.request_length_generator, PrefixRequestLengthGenerator)

        (
            hash_ids,
            num_prompt_tokens,
            num_output_tokens,
        ) = self.request_length_generator.get_next_request_params()

        prompt = '"""'
        for hash_id in hash_ids:
            if hash_id not in self.past_prompts:
                prompt_segment, num_tokens = generate_random_prompt(
                    tokenizer=self.tokenizer,
                    num_prompt_tokens=num_prompt_tokens,
                    num_output_tokens=num_output_tokens,
                    corpus_lines=self.corpus_lines,
                    add_instruction=False,
                )
                self.past_prompts[hash_id] = prompt_segment
            prompt += self.past_prompts[hash_id]

        prompt += '"""'

        prompt += (
            '\n\nINSTRUCTION: Mimic above text enclosed in """ quotes and generate '
            f"long text of at least {num_output_tokens} tokens."
        )

        default_sampling_params = {"max_tokens": num_output_tokens}
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )
        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=prompt,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=request_id,
        )

        return request_config
