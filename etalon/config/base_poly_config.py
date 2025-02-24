from abc import ABC
from dataclasses import dataclass
from typing import Any

from etalon.config.utils import get_all_subclasses


@dataclass
class BasePolyConfig(ABC):
    @classmethod
    def create_from_type(cls, type_: Any) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_type() == type_:
                return subclass()
        raise ValueError(f"Invalid type: {type_}")

    @classmethod
    def get_type(cls) -> Any:
        raise NotImplementedError(
            f"[{cls.__name__}] get_type() method is not implemented"
        )
