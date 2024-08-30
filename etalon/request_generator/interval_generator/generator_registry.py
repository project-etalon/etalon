from etalon.types.base_registry import BaseRegistry
from etalon.types.request_interval_generator_type import RequestIntervalGeneratorType

from .gamma_generator import GammaRequestIntervalGenerator
from .poisson_generator import PoissonRequestIntervalGenerator
from .static_generator import StaticRequestIntervalGenerator
from .trace_generator import TraceRequestIntervalGenerator


class RequestIntervalGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestIntervalGeneratorType:
        return RequestIntervalGeneratorType.from_str(key_str)


RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.GAMMA, GammaRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.POISSON, PoissonRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.STATIC, StaticRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.TRACE, TraceRequestIntervalGenerator
)
