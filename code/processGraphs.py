from dataGraph import DataGraph
import pickle
import random
# This functions returns the symmetric differnece of the graphs in
# filenumber.txt and filenumber+1.txt. It will choose the n vertices with
# the largest degrees and e edges at random


def getGraph(n, no_edges=-1, filenumber=1, saveToBuffer=False):
    random.seed(1)
    print("In method get graph")
    v1_list = {}
    e1_list = {}
    with open("graphData/"+str(filenumber)+".txt") as f1:
        for line in f1:
            if(line[0] == '#'):
                continue
            u = int(str.split(line)[0])
            w = int(str.split(line)[1])
            if(u not in v1_list):
                v1_list[u] = 1
            if(w not in v1_list):
                v1_list[w] = 1
            u, w = sorted([u, w])
            e1_list[(u, w)] = 1

    # print(len(v1_list))
    # print(len(e1_list))
    v2_list = {}
    e2_list = {}
    with open("graphData/"+str(filenumber+2)+".txt") as f2:
        for line in f2:
            if(line[0] == '#'):
                continue
            u = int(str.split(line)[0])
            w = int(str.split(line)[1])
            if(u not in v2_list):
                v2_list[u] = 1
            if(w not in v2_list):
                v2_list[w] = 1
            u, w = sorted([u, w])
            e2_list[(u, w)] = 1
    # print(len(v2_list))
    # print(len(e2_list))

    v_list = {}
    e_list = {}
    # Find number of common elements in v1_list and v2_list
    for v in v1_list:
        if v in v2_list:
            v_list[v] = 1

    # Need to renumber the vertices
    map = {}
    current = 0
    for v in v_list:
        map[v] = current
        current += 1
    # Get the symmetric difference of the edges
    for e1 in e1_list:
        if e1[0] not in v_list or e1[1] not in v_list:
            continue
        if e1 in e2_list:
            continue
        u, w = sorted([map[e1[0]], map[e1[1]]])
        e_list[(u, w)] = 1
    for e2 in e2_list:
        if e2[0] not in v_list or e2[1] not in v_list:
            continue
        if e2 in e1_list:
            continue
        u, w = sorted([map[e2[0]], map[e2[1]]])
        e_list[(u, w)] = 1
    print("Original Graph size")
    print(len(v_list))
    print(len(e_list))
    # Get vertex degrees
    deg = [0]*len(v_list)
    for e in e_list:
        deg[e[0]] += 1
        deg[e[1]] += 1
    # Get index of top n elements in list deg
    topV = sorted(range(len(deg)), key=lambda i: deg[i])[-n:]
    v_final = {}
    e_final = {}
    map = {}
    index = 0
    for v in topV:
        map[v] = index
        v_final[index] = 1
        index += 1
    for e in e_list:
        if e[0] not in map or e[1] not in map:
            continue
        u, w = sorted([map[e[0]], map[e[1]]])
        e_final[(u, w)] = 1
        # print(g)
    print("Before Edge Sampling")
    print(print(len(v_final), len(e_final)))
    if no_edges != -1:
        keys = random.sample(list(e_final), no_edges)
        e_final = {key: 1 for key in keys}
    print("Graph size after reduction")
    print(len(v_final), len(e_final))
    g = DataGraph(len(v_final), e_final)
    if saveToBuffer == True:
        pickle_out = open(str(filenumber)+".pkl", "wb")
        pickle.dump(g.__dict__, pickle_out)
        pickle_out.close()
    return g


if __name__ == "__main__":
    g = getGraph(20, 20, 1)
    print(g)
