from abc import ABC, abstractmethod
from typing import Tuple

from etalon.request_generator.length_generator.config import (
    BaseRequestLengthGeneratorConfig,
)


class BaseRequestLengthGenerator(ABC):
    def __init__(self, config: BaseRequestLengthGeneratorConfig):
        self.config = config

    @abstractmethod
    def get_next_num_tokens(self) -> Tuple[float, float]:
        pass
