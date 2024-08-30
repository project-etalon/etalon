import math
import random
from typing import List, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from etalon.logger import init_logger

logger = init_logger(__name__)


def generate_random_prompt(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    num_prompt_tokens: int = 1024,
    num_output_tokens: int = 128,
    corpus_lines: List[str] = None,
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
    prompt = instruction + '"""'
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
    prompt += '"""'
    return (prompt, num_prompt_tokens)
