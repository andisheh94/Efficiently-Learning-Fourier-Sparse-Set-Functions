import math
import numpy as np


class DataGraph(object):
    def __init__(self, n_v, e_list):
        self.n_v = n_v
        self.n_e = len(e_list)
        self.graph = {}
        self.shape = tuple([2]*self.n_v)
        self.sampCplx = 0
        for e in e_list:
            self.graph[e] = 1
        self.cache = {}

    def resetSampCplx(self):
        self.sampCplx = 0

    def create_dict(self):
        res = {}
        for k, w in self.graph.items():
            ntuple = [0] * self.n_v
            ntuple[k[0]] = 1
            ntuple[k[1]] = 1
            res[tuple(ntuple)] = w
        return res

    @staticmethod
    def create_from_FT(n_v, g_dict):
        g = DataGraph(n_v, {})
        try:
            g_dict.pop(tuple([0]*n_v))
        except KeyError:
            raise AssertionError("No zero Frequency")
        for k, w in g_dict.items():
            k = np.array(k)
            if np.sum(k) != 2:
                raise AssertionError(f"Invalid entry {k}:{w}!")
            a, b = k.nonzero()[0]
            g.add_edge(a, b, w * (-2))
        return g

    @staticmethod
    def create_from_FT2(n_v, g_dict):
        g = DataGraph(n_v, {})
        try:
            g_dict.pop(tuple([0]*n_v))
        except KeyError:
            pass
        for k, w in g_dict.items():
            k = np.array(k)
            if np.sum(k) != 2:
                continue
            a, b = k.nonzero()[0]
            g.add_edge(a, b, w * (-2))
        return g

    def add_edge(self, a, b, w):
        a, b = sorted([a, b])
        self.graph[(a, b)] = w
        self.n_e += 1

    # Overloading[] operator
    def __getitem__(self, key):
        try:
            return self.cache[key]
        except KeyError:
            pass
        self.sampCplx += 1
        res = 0.0
        for edge, w in self.graph.items():
            if key[edge[0]] != key[edge[1]]:
                res += w
        self.cache[key] = res
        return res

    # Function for prinitng the graph
    def __str__(self):
        return str(self.graph)

    def reconstError(self, other):
        sum = 0
        for k in self.graph:
            if k not in other.graph:
                sum += 1
        for k in other.graph:
            if k not in self.graph:
                sum += 1

        return sum

    def __eq__(self, other):
        count = 0
        for k, w in self.graph.items():
            try:
                if math.isclose(w, other.graph[k], rel_tol=0.0001):
                    count += 1
            except KeyError:
                return False
        return count == len(other.graph)
