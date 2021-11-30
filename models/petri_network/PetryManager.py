from typing import Union

import numpy as nmp
from igraph import Graph, plot
import pandas

from PetryNetwork import PetryNetwork


class PetryManager:

    def __init__(self, p: int, t: int, o: nmp.ndarray, i: nmp.ndarray, m_0: nmp.ndarray, save_path: str):
        super().__init__()
        self.save_path = save_path
        self.o_matrix = o
        self.i_matrix = i
        self.m = m_0
        self.p = p
        self.t = t
        self.networks: list[PetryNetwork] = []
        self.m_dict = {}
        self.m_map = {}
        self.stops = set()
        self.stable = True
        self.is_limited_exit = False

    def __expand_m_dict(self, key):
        self.m_dict[key] = set()
        self.m_map[key] = len(self.m_map.keys())

    def safe_path(self, path_t: Union[nmp.int64, nmp.ndarray], path_end_state: tuple[int],
                  path_start_state: tuple[int]) -> bool:
        if type(path_t) == nmp.int64:
            t = (path_t,)
        else:
            t = tuple(path_t)
        to_add = (t, path_end_state)
        res = to_add not in self.m_dict[path_start_state]
        self.m_dict[path_start_state].add(to_add)
        if path_end_state not in self.m_dict.keys():
            self.__expand_m_dict(path_end_state)
            return True
        return res

    def get_transactions(self, petry: PetryNetwork) -> nmp.ndarray:
        res = []
        for i in range(self.t):
            if petry.check_transaction(i):
                res.append(i)
        return nmp.array(res)

    def __reset(self):
        print(f'M0: {self.m}')
        self.networks = [PetryNetwork(self.p, self.t, self.o_matrix, self.i_matrix, self.m)]
        self.m_map = {}
        self.__expand_m_dict(tuple(self.m))
        self.stable = True
        self.stops = set()
        self.is_limited_exit = False
        self.networks[0].plot_and_safe(f'{self.save_path}/start.png')

    def run_steps(self, count: int):
        self.__reset()
        for i in range(count):
            if not self.calc_next_possible():
                print(f'Stopped  at step {i} due to no actions')
                break

    def run_pausing(self):
        self.__reset()
        while self.calc_next_possible():
            c = input()
            if c == 'exit':
                break

    def run(self):
        self.__reset()
        print('TBD')

    def calc_next_possible(self) -> bool:
        i = 0
        is_transacted = False
        limit = len(self.networks)
        while i < limit:
            network = self.networks[i]
            transactions = network.get_possible_transactions()
            if len(transactions) == 1:
                is_transacted = is_transacted | self.__run_transaction(network, transactions[0])
            elif len(transactions) > 1:
                ways = network.get_compatible_transactions()
                if len(ways) == 1:
                    is_transacted = is_transacted | self.__run_transaction(network, ways[0])
                elif len(ways) > 1:
                    self.stable = False
                    new_networks = []
                    self.networks.remove(network)
                    i -= 1
                    limit -= 1
                    for j, (w) in enumerate(ways):
                        nn = PetryNetwork(
                            network.p,
                            network.t,
                            network.i_matrix,
                            network.o_matrix,
                            network.m.copy()
                        )
                        new_networks.append(nn)
                        is_transacted = is_transacted | self.__run_transaction(nn, w)
                    self.networks += new_networks
            else:
                i -= 1
                limit -= 1
                self.stops.add(tuple(network.m))
                self.networks.remove(network)
            i += 1

        if not is_transacted:
            self.is_limited_exit = True
        return is_transacted

    def __run_transaction(self, network: PetryNetwork, t: Union[nmp.int64, nmp.ndarray]) -> bool:
        start_state = tuple(network.m)
        network.perform_transaction(t)
        m_tuple = tuple(network.m)
        was_path_updated = self.safe_path(t + 1, m_tuple, start_state)
        name = self.m_map[m_tuple]
        print(f'M{name}: {network.m} by {t + 1} from M{self.m_map[start_state]}')
        network.plot_and_safe(f'{self.save_path}/{name}.png')
        return was_path_updated

    def plot_m_diagram(self):
        m_indices_dict = {}
        for i, (m_i) in enumerate(self.m_dict.keys()):
            m_indices_dict[m_i] = i
        g = Graph(directed=True)
        g.add_vertices(len(m_indices_dict.keys()))
        e_labels = []
        for m in self.m_dict:
            local_edges_dict = {}
            for (t, start) in self.m_dict[m]:
                if start in local_edges_dict.keys():
                    local_edges_dict[start] += f'\n{t}'
                else:
                    local_edges_dict[start] = str(t)

            for start in local_edges_dict.keys():
                g.add_edge(m_indices_dict[m], m_indices_dict[start])
            e_labels += list(local_edges_dict.values())

        layout = g.layout('kk')
        g.es['label'] = e_labels
        g.es['color'] = 'gray'
        g.vs['label'] = [f'M{i}={m}' for i, (m) in enumerate(self.m_dict.keys())]
        g.vs['color'] = 'white'
        g.vs['frame_color'] = 'white'
        g.vs['size'] = 40
        visual_style = {
            'margin': 100,
        }

        plot(g, f'{self.save_path}/m.png', layout=layout,
             **visual_style)

    def __check_limited(self) -> int:
        if self.is_limited_exit:
            return nmp.array(list(self.m_dict.keys())).max(initial=0)
        else:
            return -nmp.array(list(self.m_dict.keys())).max(initial=0)

    def __check_secure(self, limited: int) -> bool:
        return limited == 1

    def __check_permanent_load(self):
        return nmp.unique(nmp.array(list(self.m_dict.keys())).sum(axis=1))

    def __check_live(self):
        if len(self.stops) == 0:
            return True
        else:
            return f'False, {self.stops}'

    def get_run_characteristics(self) -> dict:
        limited = self.__check_limited()
        load = self.__check_permanent_load()
        return {
            'limited': f'{limited >= 0}, {abs(limited)}',
            'secure': self.__check_secure(limited),
            '1-conservative': f'{len(load) == 1}, {load}',
            'live': self.__check_live(),
            'achivable m': list(self.m_dict.keys()),
            'stable': self.stable
        }

    def __check_is_free_choice(self):
        res = set()
        for p_i in range(self.p):
            i_sum = self.o_matrix[p_i].sum()
            if i_sum > 1:
                for t_i in range(self.t):
                    if self.o_matrix[p_i, t_i] > 0 and self.o_matrix[:, t_i].sum() > 1:
                        res.add(p_i)
                        break
        if len(res) == 0:
            return True
        else:
            return f'False {list(res)}'

    def __check_marked_graph(self):
        p = nmp.arange(self.p) + 1
        mask = nmp.logical_or(self.i_matrix.sum(axis=1) != 1, self.o_matrix.sum(axis=1) != 1)
        p = p[mask]
        if len(p) == 0:
            return True
        else:
            return f'False {p}'

    def __check_automate(self):
        t = nmp.arange(self.t) + 1
        mask = nmp.logical_or(self.i_matrix.sum(axis=0) > 1, self.o_matrix.sum(axis=0) > 1)
        t = t[mask]
        if len(t) == 0:
            return True
        else:
            return f'False {t}'

    def __check_io_correspond(self, p_i):
        for i in range(self.t):
            if self.i_matrix[p_i, i] == 1 and self.o_matrix[p_i, i] == 0:
                return False
        return True

    def __check_no_conflicts(self):
        r = []
        for p_i in range(self.p):
            if self.i_matrix[p_i].sum() > 1 and not self.__check_io_correspond(p_i):
                r.append(p_i + 1)
        if len(r) == 0:
            return True
        else:
            return f'False {r}'

    def get_scheme_characteristics(self) -> dict:
        return {
            'free choice net': self.__check_is_free_choice(),
            'marked graph': self.__check_marked_graph(),
            'automate': self.__check_automate(),
            'no conflicts': self.__check_no_conflicts()
        }


def print_dict(d: dict):
    for (k, v) in d.items():
        print(f'\'{k}\': {v}')


def main(i: nmp.ndarray, o: nmp.ndarray, m_0: nmp.ndarray, save_path: str):
    p, t = i.shape
    manager = PetryManager(p, t, o, i, m_0, save_path)
    print_dict(manager.get_scheme_characteristics())
    manager.run_steps(20)
    manager.plot_m_diagram()
    print_dict(manager.get_run_characteristics())


default_save_path = '/home/hornedheck/PycharmProjects/scientificProject/data/petry_plots'


def test_data():
    o = nmp.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])
    i = nmp.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    m = nmp.array([2, 1, 0, 0, 0, 1, 0])
    main(i, o, m, default_save_path)


def read_data():
    i_matrix = nmp.array(pandas.read_csv(input('Enter i matrix path (.csv):'), header=None))
    o_matrix = nmp.array(pandas.read_csv(input('Enter o matrix path (.csv):'), header=None))
    m = nmp.array([int(s) for s in input('Enter M0 in format: u_1 u_2 ...:').split(' ')])
    save_path = input('Enter save path:')
    main(i_matrix, o_matrix, m, save_path)


if __name__ == '__main__':
    read_data()
