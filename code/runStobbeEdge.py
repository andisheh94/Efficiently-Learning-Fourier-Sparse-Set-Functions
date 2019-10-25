import numpy as np
from sparseWHT import SWHT
from dataGraph import DataGraph
from processGraphs import getGraph
from randomGraph import Graph
import pickle
import time
from proximalMethod import ProximalMethod
import statistics
if __name__ == "__main__":
    np.random.seed(1)

    iter = 20
 # Stobbe's method
 # No. of vertices fixed No. of edges changes
    n = 100
    for filenum in [1, 3, 5, 7]:
        info_e = {}
        for no_edges in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
            g = getGraph(n, no_edges, filenum)
            for C in np.arange(1.5, 3.4, 0.5):
                print("no_edges=", no_edges, "C=", C)
                no_success = 0
                prox = ProximalMethod(n, no_edges, C)
                sumTime = 0
                listTime = []
                totalSampCplx = 0
                for j in range(iter):
                    now = time.time()
                    y = prox.run(g, 1.0)
                    duration = time.time()-now
                    sumTime += duration
                    listTime.append(duration)
                    try:
                        g2 = DataGraph.create_from_FT(prox.n, y)
                        if g == g2:
                            no_success += 1
                    except AssertionError:
                        pass
                    totalSampCplx += g.sampCplx
                    g.cache = {}
                    g.resetSampCplx()
                print("no_succes=", no_success/iter, "sampCplx=",
                      totalSampCplx/iter, "time=", sumTime/iter, "median", statistics.median(listTime), listTime)
                try:
                    info_e[no_edges].append((no_success/iter, totalSampCplx/iter,
                                             sumTime/iter, statistics.median(listTime), listTime))
                except KeyError:
                    info_e[no_edges] = []
                    info_e[no_edges].append((no_success/iter, totalSampCplx/iter,
                                             sumTime/iter, statistics.median(listTime), listTime))
            print(info_e[no_edges])
        print(info_e)
        pickle_out = open("experimentResults/StobbeEdgeFileSec"+str(filenum)+".pkl", "wb")
        pickle.dump(info_e, pickle_out)
        pickle_out.close()
