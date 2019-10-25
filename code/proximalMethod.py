import apgpy as apg
import numpy as np
from dataGraph import DataGraph
import pickle
from math import log, ceil, isclose
from processGraphs import getGraph
from randomGraph import Graph


class ProximalMethod():
    def __init__(self, n, K, C):
        self.n = n
        self.K = K
        self.C = C

    # def load_graph(self):
    #     g = DataGraph(0, {})
    #     with open('GraphData/1.pkl', 'rb') as f:
    #         d = pickle.load(f)
    #         g.graph = d["graph"]
    #         g.n_v = d["n_v"]
    #         g.n_e = d["n_e"]
    #         g.shape = d["shape"]
    #     return g

    def grad(self, x):
        # grad = 2 \psi^T (\psi x - y)
        X = x.reshape((self.n, self.n))
        # Apply psi
        out = np.dot(np.dot(self.psi, X), self.psi.transpose())
        out = np.diag(out) - self.y
        out = np.diag(out)
        out = np.dot(np.dot(self.psi.transpose(), out), self.psi)
        out = out.reshape((-1,))
        return out / self.lmda
        # apply psi^T

    def proximal(self, v, t):
        return np.sign(v) * np.maximum(abs(v) - t, 0)

    def sampleGraph(self, g):
        # np.random.seed(1)
        # m = No. of measurements
        self.m = int(self.C * ceil((self.K)*log(self.n, 2)))
        self.psi = np.zeros((self.m, self.n))
        self.y = np.zeros(self.m)
        dict = {}
        for j in range(self.m):
            # Generate a random cut
            cut = [np.random.randint(0, 2) for _ in range(self.n)]
            while tuple(cut) in dict:
                cut = [np.random.randint(0, 2) for _ in range(self.n)]
            dict[tuple(cut)] = 1
            #cut = [0] * self.n
            # for i in range(self.n):
            #cut[i] = np.random.randint(0, 2)
            self.psi[j, :] = (-1) ** np.array(cut)
            self.y[j] = g[tuple(cut)]
        #print("sampling finished")

    def run(self, g, lmda=1.0):
        self.sampleGraph(g)
        self.lmda = lmda
        x = apg.solve(self.grad, self.proximal, np.zeros(self.n ** 2), quiet=True)
        return self.getFourTrans(x)

    def getFourTrans(self, x):
        x = x.reshape(self.n, self.n)
        fourier = {}
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if not isclose(x[i, j], 0, abs_tol=0.0001):
                    freq = [0]*self.n
                    freq[i] = freq[j] = 1
                    fourier[tuple(freq)] = -0.5
        fourier[tuple([0]*self.n)] = 1
        return fourier


if __name__ == "__main__":
    n = 100
    K = 295
    g = Graph(n, K)
    prox = ProximalMethod(n, K, 2.2)
    print(g)
    p = 0
    for i in range(1):
        out = prox.run(g, 1)
        try:
            g2 = Graph.create_from_FT(prox.n, out)
        except AssertionError:
            g.cache = {}
            continue
        if g == g2:
            p = p+1
        g.cache = {}
    print(p)
    print(g.sampCplx)
