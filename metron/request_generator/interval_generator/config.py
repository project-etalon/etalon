from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseRequestIntervalGeneratorConfig:
    seed: Optional[int] = None


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    cv: Optional[float] = None
    qps: Optional[float] = None


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    pass


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: Optional[float] = None


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    time_scale_factor: Optional[float] = None
