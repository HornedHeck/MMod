import math

from models.brv.base_random_value import BaseRandomValue
import numpy as nmp

MAX_VALUE = 65536


def r_test(
        brv: BaseRandomValue,
        n: nmp.ndarray = nmp.array([100, 10000, 1000000, 100000000]),
        s: int = 50
) -> nmp.ndarray:
    return nmp.array([__r_test_int__(brv, n_i, s) for n_i in n])


def __r_test_int__(
        brv: BaseRandomValue,
        n: int,
        s: int
) -> float:
    data = nmp.array([brv.next() for i in range(n + s)]) / MAX_VALUE
    x_data: nmp.ndarray = data[0:n]
    y_data: nmp.ndarray = data[s:]
    e_x = nmp.mean(x_data)
    e_y = nmp.mean(y_data)
    e_xy = nmp.mean(x_data * y_data)
    d_x = nmp.var(x_data)
    d_y = nmp.var(y_data)
    return (e_xy - e_x * e_y) / math.sqrt(d_x * d_y)
