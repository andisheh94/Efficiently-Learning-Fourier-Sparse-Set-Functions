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
    # No. of edges fixed No. of vertices changes
    no_edges = 50
    for filenum in [1, 3, 5, 7]:
        info_n = {}
        if filenum == 5:
            l = [70, 90, 110, 130, 150, 170, 190, 210]
        else:
            l = [50, 70, 90, 110, 130, 150, 170, 190, 210]
        for n in l:
            g = getGraph(n, no_edges, filenum)
            for C in np.arange(1.5, 3.4, 0.5):
                print("n=", n, "C=", C)
                #print("C=", C)
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
                    info_n[n].append((no_success/iter, totalSampCplx/iter,
                                      sumTime/iter, statistics.median(listTime), listTime))
                except KeyError:
                    info_n[n] = []
                    info_n[n].append((no_success/iter, totalSampCplx/iter,
                                      sumTime/iter, statistics.median(listTime), listTime))
            print(info_n[n])
        print(info_n)
        pickle_out = open("experimentResults/StobbeVertexFileSec"+str(filenum)+".pkl", "wb")
        pickle.dump(info_n, pickle_out)
        pickle_out.close()
