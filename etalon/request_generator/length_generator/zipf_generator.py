from typing import Tuple

import numpy as np

from etalon.request_generator.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)

EPS = 1e-8


class ZipfGenerator:
    def __init__(
        self, min: int, max: int, theta: float, scramble: bool, seed: int
    ) -> None:
        self.min = min
        self.max = max
        self.items = max - min + 1
        self.theta = theta
        self.zeta_2 = self._zeta(2, self.theta)
        self.alpha = 1.0 / (1.0 - self.theta)
        self.zetan = self._zeta(self.items, self.theta)
        self.eta = (1 - np.power(2.0 / self.items, 1 - self.theta)) / (
            1 - self.zeta_2 / (self.zetan + EPS)
        )
        self.scramble = scramble
        self.seed = seed
        self.generator = np.random.RandomState(seed)

    def _zeta(self, count: float, theta: float) -> float:
        return np.sum(1 / (np.power(np.arange(1, count), theta)))

    def _next(self) -> int:
        u = self.generator.random_sample()
        uz = u * self.zetan

        if uz < 1.0:
            return self.min

        if uz < 1.0 + np.power(0.5, self.theta):
            return self.min + 1

        return self.min + int(
            (self.items) * np.power(self.eta * u - self.eta + 1, self.alpha)
        )

    def next(self) -> int:
        retval = self._next()
        if self.scramble:
            retval = self.min + hash(str(retval) + str(self.seed)) % self.items

        return retval


class ZipfRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._zipf_generator = ZipfGenerator(
            self.config.min_tokens,
            self.config.max_tokens,
            self.config.theta,
            self.config.scramble,
            self.config.seed,
        )

    def get_next_num_tokens(self) -> Tuple[float, float]:
        total_tokens = self._zipf_generator.next()

        decode_tokens = total_tokens / (1 + self.config.prefill_to_decode_ratio)
        prefill_tokens = total_tokens - decode_tokens

        return prefill_tokens, decode_tokens
