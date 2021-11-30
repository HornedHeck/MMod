import random

from models.brv.base_random_value import BaseRandomValue

MIN_SEED = 128
MAX_SEED = 255
L_CUT = 4
R_CUT = 256


class BiSquareMiddleBRV(BaseRandomValue):

    def __init__(self) -> None:
        super().__init__()
        self.seed1 = random.randint(MIN_SEED, MAX_SEED)
        self.seed2 = random.randint(MIN_SEED, MAX_SEED)

    def next(self) -> int:
        self.seed1 = (self.seed1 * self.seed1) // L_CUT % R_CUT
        self.seed2 = (self.seed2 * self.seed2) // L_CUT % R_CUT
        return self.seed1 * 256 + self.seed2
