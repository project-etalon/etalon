from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseRequestLengthGeneratorConfig:
    seed: Optional[int] = None


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: Optional[int] = None
    decode_tokens: Optional[int] = None


@dataclass
class SyntheticRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    prefill_to_decode_ratio: Optional[float] = None


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: Optional[float] = None
    scramble: Optional[bool] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    prefill_to_decode_ratio: Optional[float] = None


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: Optional[str] = None
    prefill_scale_factor: Optional[float] = None
    decode_scale_factor: Optional[float] = None
    max_tokens: Optional[int] = None
