import math
import numpy as np


class Graph(object):
    def __init__(self, n_v, n_e, max_w=100.0):
        self.n_v = n_v
        self.n_e = 0
        self.max_w = max_w
        self.graph = {}
        self.shape = tuple([2]*n_v)
        self.sampCplx = 0
        self.cache = {}
        for i in range(n_e):
            self.add_random_edge()

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
        g = Graph(n_v, 0)
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

    def add_random_edge(self):
        a = 0
        b = 0
        while a == b or (a, b) in self.graph:
            a = np.random.randint(self.n_v)
            b = np.random.randint(self.n_v)
            a, b = sorted([a, b])
        w = np.random.uniform(low=0, high=self.max_w)
        self.add_edge(a, b, 1)

    def statistic(self, detFreq):
        correct = 0
        incorrect = 0
        for key in detFreq:
            freq = np.array(key)
            index = np.nonzero(freq)
            if index[0].shape == (2,):
                index = tuple(index[0])
                if index in self.graph and detFreq[key] == -0.5:
                    print(index, "Correct")
                    correct += 1
                else:
                    print(index, "InCorrect")
                    incorrect += 1
            elif index[0].shape == (0,):
                if detFreq[key] == self.n_e/2:
                    correct += 1
                    print("0 freq correct")
                else:
                    incorrect += 1
                    print("0 freq incorrect")
            else:
                incorrect += 1
        print("Correct=", correct, "Incorrect=", incorrect, "total=", correct+incorrect)
        return correct, incorrect

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
        res = 0.0
        self.sampCplx += 1
        for edge, w in self.graph.items():
            if key[edge[0]] != key[edge[1]]:
                res += w
        self.cache[key] = res
        return res

    # Function for prinitng the graph
    def __str__(self):
        return str(self.graph)

    def __eq__(self, other):
        count = 0
        for k, w in self.graph.items():
            try:
                if math.isclose(w, other.graph[k], rel_tol=0.0001):
                    count += 1
            except KeyError:
                #print("Error in k", k)
                return False
        return count == len(other.graph)


if __name__ == '__main__':
    g = Graph(25, 5)
    print(g)
    cut = np.random.randint(0, high=2, size=25)
    print(cut)
    print(g[cut])
    print()

    g2 = Graph.create_from_FT(25, g.create_dict())
    print(g == g2)
