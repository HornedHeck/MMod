from models.brv.base_random_value import BaseRandomValue
import numpy as nmp

MAX_VALUE = 65536


def frequency_test(
        brv: BaseRandomValue,
        k: int = 10,
        n1: int = 100,
        n2: int = 10_000
) -> tuple[nmp.ndarray, nmp.ndarray]:
    res1 = __frequency_test_int(brv, k, n1)
    res2 = __frequency_test_int(brv, k, n2)
    return res1, res2


def __frequency_test_int(brv: BaseRandomValue, k: int, n: int) -> nmp.ndarray:
    src = nmp.array([brv.next() for i in range(n)])
    step = MAX_VALUE / k
    groups = nmp.array([int((i + 1) * step) for i in range(k)])
    src = nmp.array(
        [nmp.argmin(x > groups) for x in src]
    )
    v, c = nmp.unique(src, return_counts=True)
    res = nmp.zeros(k, dtype=float)
    res[v] = c / n
    return res