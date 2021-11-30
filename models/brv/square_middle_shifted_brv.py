import random

from models.brv.base_random_value import BaseRandomValue

MIN_SEED = 33000
MAX_SEED = 60000
L_CUT = 16
R_CUT = 65535


class ShiftedSquareMiddleBRV(BaseRandomValue):

    def __init__(self) -> None:
        super().__init__()
        self.seed = random.randint(MIN_SEED, MAX_SEED)

    def next(self) -> int:
        self.seed = ((self.seed + 7) * (self.seed + 2)) // L_CUT % R_CUT
        return self.seed
