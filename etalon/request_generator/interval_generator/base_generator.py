from abc import ABC, abstractmethod

from etalon.request_generator.interval_generator.config import (
    BaseRequestIntervalGeneratorConfig,
)


class BaseRequestIntervalGenerator(ABC):
    def __init__(self, config: BaseRequestIntervalGeneratorConfig):
        self.config = config

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass
