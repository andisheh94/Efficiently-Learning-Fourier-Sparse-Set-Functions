import numpy as np
from sparseWHT import SWHT
from dataGraph import DataGraph
from processGraphs import getGraph
from randomGraph import Graph
import pickle
import time
from proximalMethod import ProximalMethod

if __name__ == "__main__":
    # Reconstruction Error Experiment
    n = 70
    # Stobbe Method
    reconstError = {}
    for filenum in [1, 3, 5, 7]:
        g = getGraph(n, -1, filenum)
        # print(g)
        no_edges = g.n_e
        for C in np.arange(0.62, 2.2, 0.02):
            print("C=", C)
            np.random.seed(1)
            prox = ProximalMethod(n, no_edges, C)
            y = prox.run(g, 1.0)
            g2 = DataGraph.create_from_FT2(prox.n, y)
            # print(g2)
            reconstError[(filenum, C)] = (g.reconstError(g2), g.sampCplx)
            print(g.reconstError(g2), g.sampCplx)
            g.resetSampCplx()
            g.cache = {}

    # print(reconstError)
    pickle_out = open("experimentResults/reconstructStobbe2.pkl", "wb")
    pickle.dump(n, pickle_out)
    pickle.dump(no_edges, pickle_out)
    pickle.dump(reconstError, pickle_out)
    pickle_out.close()

    # SHWT
    reconstError = {}
    for filenum in [1, 3, 5, 7]:
        g = getGraph(n, -1, filenum)
        # print(g)
        no_edges = g.n_e
        for C in np.arange(0.8, 1.5, 0.1):
            for ratio in np.arange(1.1, 2, 0.05):
                print("C=", C, "ratio=", ratio)
                swht = SWHT(n, no_edges, C, ratio, 3, 1)
                y = swht.run(g)
                g2 = DataGraph.create_from_FT2(swht.n, y)
                # print(g2)
                reconstError[(filenum, C, ratio)] = (g.reconstError(g2), g.sampCplx)
                print(g.reconstError(g2), g.sampCplx)
                g.resetSampCplx()
                g.cache = {}
    print(reconstError)
    pickle_out = open("experimentResults/reconstructSHWT2.pkl", "wb")
    pickle.dump(n, pickle_out)
    pickle.dump(no_edges, pickle_out)
    pickle.dump(reconstError, pickle_out)
    pickle_out.close()
