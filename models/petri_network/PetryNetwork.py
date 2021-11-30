from enum import Enum
from typing import Union

import numpy as nmp
from igraph import Graph
from igraph import plot


class VType(Enum):
    P = 0
    T = 1


def has_bigger_parents(t: nmp.ndarray, approved_t: list[nmp.ndarray]) -> bool:
    for a_t in approved_t:
        if all(nmp.isin(t, a_t)):
            return True
    return False


def all_combinations(values: nmp.ndarray):
    raw = nmp.array(nmp.meshgrid(*[values for _ in range(len(values))])).T.reshape(-1, len(values))
    un = {tuple(nmp.unique(r)) for r in raw}
    arrays = nmp.array([nmp.array(u) for u in un], dtype=object)
    idx = nmp.argsort([len(a) for a in arrays])[::-1]
    return arrays[idx]


class PetryNetwork:
    colors = {
        VType.P: 'white',
        VType.T: 'yellow'
    }

    def __init__(self, p: int, t: int, i: nmp.ndarray, o: nmp.ndarray, m_0: nmp.ndarray):
        super().__init__()
        self.t = t
        self.p = p
        self.i_matrix = i
        self.o_matrix = o
        self.m = m_0

    def check_transaction(self, t_i: int) -> bool:
        p_map = self.o_matrix[:, t_i] > 0
        return all(self.m[p_map] > 0)

    def __perform_t(self, t_i: int):
        o_map = self.o_matrix[:, t_i] > 0
        i_map = self.i_matrix[:, t_i] > 0
        self.m[o_map] -= 1
        self.m[i_map] += 1

    def perform_transaction(self, t: Union[nmp.int64, nmp.ndarray]):
        if type(t) is nmp.int64:
            self.__perform_t(t)
        else:
            for i in t:
                self.__perform_t(i)

    def plot_and_safe(self, name: str):
        g = Graph(directed=True)
        g.add_vertices(self.p + self.t)
        g.vs["vtype"] = [VType.P for _ in range(self.p)] + [VType.T for _ in range(self.t)]
        for i in range(self.p):
            for j in range(self.t):
                if self.i_matrix[i, j] > 0:
                    g.add_edge(self.p + j, i)
                if self.o_matrix[i, j] > 0:
                    g.add_edge(i, self.p + j)

        layout = g.layout('kk')

        visual_style = {
            'vertex_color': [self.colors[t] for t in g.vs['vtype']],
            'vertex_label': [f'p{p + 1}({self.m[p]})' for p in range(self.p)] + [f't{t + 1}' for t in range(self.t)],
            'vertex_size': 45
        }

        plot(g, name, layout=layout, **visual_style)

    def get_possible_transactions(self) -> nmp.ndarray:
        res = []
        for i in range(self.t):
            if self.check_transaction(i):
                res.append(i)
        return nmp.array(res)

    def get_compatible_transactions(self) -> nmp.ndarray:
        comb = all_combinations(self.get_possible_transactions())
        res = []
        for c in comb:
            if self.check_compatible_transactions(c) and not has_bigger_parents(c, res):
                res.append(c)
        return nmp.array(res)

    def check_compatible_transactions(self, t: nmp.ndarray):
        t_o: nmp.ndarray = self.o_matrix[:, t].sum(axis=1)
        return all(t_o <= self.m)
