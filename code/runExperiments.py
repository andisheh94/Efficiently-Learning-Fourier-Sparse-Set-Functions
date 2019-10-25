import numpy as np
from sparseWHT import SWHT
from dataGraph import DataGraph
from processGraphs import getGraph
from randomGraph import Graph
import pickle
import time
from proximalMethod import ProximalMethod

if __name__ == "__main__":
    iter = 10
# Our Method
# No. of vertices fixed No. of edges changes
    n = 100
    info_e = {}
    for no_edges in [50, 75, 100, 125, 150, 175, 200]:
        g = getGraph(n, no_edges, 1)
        for C in np.arange(0.7, 2, 0.05):
            for ratio in np.arange(1.2, 3, 0.02):
                print("no_edges=", no_edges, "C=", C, "ratio=", ratio)
                #print("C=", C)
                no_success = 0
                swht = SWHT(n, no_edges, C, ratio, 3, 1)
                sumTime = 0
                totalSampCplx = 0
                for j in range(iter):
                    now = time.time()
                    y = swht.run(g)
                    duration = time.time()-now
                    sumTime += duration
                    try:
                        g2 = Graph.create_from_FT(swht.n, y)
                        if g == g2:
                            no_success += 1
                    except AssertionError:
                        pass
                    totalSampCplx += g.sampCplx
                    g.cache = {}
                    g.resetSampCplx()
                #print("no_succes=", no_success, "sampCplx=", g.sampCplx)
                try:
                    info_e[no_edges].append(
                        (no_success/iter, totalSampCplx/iter, sumTime/iter, C, ratio))
                except KeyError:
                    info_e[no_edges] = []
                    info_e[no_edges].append(
                        (no_success/iter, totalSampCplx/iter, sumTime/iter, C, ratio))
            g.resetSampCplx()
            # print(info_e[no_edges])
        print(info_e[no_edges])
    print(info_e)
    pickle_out = open("experimentResults/SHWTEdge1.pkl", "wb")
    pickle.dump(info_e, pickle_out)
    pickle_out.close()
    # No. of edges fixed No. of vertices changes
    no_edges = 20
    info_n = {}
    for n in [20, 30, 40, 50, 80, 100, 200, 400, 800, 1000]:
        g = getGraph(n, no_edges, 1)
        for C in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1, 2, 4]:
            print("n", n, "C=", C)
            #print("C=", C)
            no_success = 0
            swht = SWHT(n, no_edges, C)
            sumTime = 0
            for j in range(iter):
                now = time.time()
                y = swht.run(g)
                duration = time.time()-now
                sumTime += duration
                try:
                    g2 = Graph.create_from_FT(swht.n, y)
                    if g == g2:
                        no_success += 1
                except AssertionError:
                    pass
            #print("no_succes=", no_success, "sampCplx=", g.sampCplx)
            try:
                info_n[n].append((no_success/iter, g.sampCplx, sumTime/iter))
            except KeyError:
                info_n[n] = []
                info_n[n].append((no_success/iter, g.sampCplx, sumTime/iter))
            g.resetSampCplx()
        print(info_n[n])
    print(info_n)
    pickle_out = open("ourMethod1.pkl", "wb")
    pickle.dump(info_e, pickle_out)
    pickle.dump(info_n, pickle_out)
    pickle_out.close()
