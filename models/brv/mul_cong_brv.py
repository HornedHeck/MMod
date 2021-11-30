import random

from models.brv.base_random_value import BaseRandomValue

MIN_SEED = 16384
MAX_SEED = 32767
M = 65536
K = 27253


class MulCongBRV(BaseRandomValue):

    def __init__(self, k: int = K) -> None:
        super().__init__()
        self.k = k
        self.seed = random.randint(MIN_SEED, MAX_SEED)

    def next(self) -> int:
        self.seed = self.k * self.seed % M
        return self.seed
