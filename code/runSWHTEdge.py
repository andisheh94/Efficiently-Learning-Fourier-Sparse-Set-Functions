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
    iter = 20
# Our Method
# No. of vertices fixed No. of edges changes
    n = 100
    for filenum in [1, 3, 5, 7]:
        info_e = {}
        for no_edges in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
            g = getGraph(n, no_edges, filenum)
            for C in [1.1, 1.2]:
                for ratio in [1.4, 1.3]:
                    print("no_edges=", no_edges, "C=", C, "ratio=", ratio)
                    #print("C=", C)
                    no_success = 0
                    swht = SWHT(n, no_edges, C, ratio, 3, 1)
                    sumTime = 0
                    listTime = []
                    totalSampCplx = 0
                    for j in range(iter):
                        now = time.time()
                        y = swht.run(g)
                        duration = time.time()-now
                        sumTime += duration
                        listTime.append(duration)
                        try:
                            g2 = Graph.create_from_FT(swht.n, y)
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
                        info_e[no_edges].append(
                            (no_success/iter, totalSampCplx/iter, sumTime/iter, C, ratio, statistics.median(listTime), listTime))
                    except KeyError:
                        info_e[no_edges] = []
                        info_e[no_edges].append(
                            (no_success/iter, totalSampCplx/iter, sumTime/iter, C, ratio, statistics.median(listTime), listTime))
                g.resetSampCplx()
                # print(info_e[no_edges])
            print(info_e[no_edges])
        print(info_e)
        pickle_out = open("experimentResults/SWHTEdgeFileSec"+str(filenum)+".pkl", "wb")
        pickle.dump(info_e, pickle_out)
        pickle_out.close()
