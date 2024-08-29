from etalon.request_generator.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)


class StaticRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def get_next_inter_request_time(self) -> float:
        return 0
