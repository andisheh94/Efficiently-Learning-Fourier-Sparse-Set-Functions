from math import log, ceil
import numpy as np
from math import isclose
from graphData.DataGraph import DataGraph
import pickle
from randomGraph import Graph

# Takes i and gives coreesponding frequncey of this coordinate


def getFreq(coor, n):
    i = int(coor/n)
    j = coor % n
    if(i == j and i == 0):
        return [0]*n
    if(i < j):
        temp = [0]*n
        temp[i] = 1
        temp[j] = 1
        return temp
    return None


def MatchingPursuit(measurement, n, m):
    b = np.zeros(m)
    for j in range(m):
        b[j] = measurement[j][1]
    # residue vector
    r = np.array(b)
    #print("b=", b, "r=", r)
    current_guess = {}
    amme = 0
    while not isclose(np.linalg.norm(r, np.inf), 0, abs_tol=0.001):
        amme += 1
        inp = {}
        #print("Getting inner Products")
        for coor in range(n ** 2):
            freq = getFreq(coor, n)
            if freq == None:
                continue

            inp[coor] = np.abs(np.dot(r, getHadamardVector(freq, measurement)))
            # if amme == 2:
            #     print("Freq=", freq, "inp=", inp[coor])
        # Get largest inner product
        argmax = max(inp, key=inp.get)
        current_guess[argmax] = 1
        # DO projection
        #print(m, len(current_guess))
        X = np.zeros((m, len(current_guess)))
        # print(X.shape)
        i = 0
        #print("Current Guess is", current_guess)
        for coor in current_guess:
            freq = getFreq(coor, n)
            X[:, i] = getHadamardVector(freq, measurement)
            i += 1
        # (X^T X)^{-1}
        middleTerm = np.linalg.inv(np.dot(X.transpose(), X))
        # X^Tb
        rightTerm = np.dot(X.transpose(), b)
        # projection
        P = np.dot(X, np.dot(middleTerm, rightTerm))
        r = b - P
        #print("X=", X, "b=", b, "r=", r, "P=", P)
# Gives the column of the ovservation matrix coressponding to freq
    return current_guess


def getHadamardVector(freq, measurement):
    out = np.zeros(m)
    for j in range(m):
        cut = measurement[j][0]
        inp = int(np.dot(np.array(freq), np.array(cut)))
        out[j] = (-1) ** inp
    #out = out/np.sqrt(m)
    return out


if __name__ == "__main__":
    g = DataGraph(0, {})
    with open('GraphData/1.pkl', 'rb') as f:
        d = pickle.load(f)
        g.graph = d["graph"]
        g.n_v = d["n_v"]
        g.n_e = d["n_e"]
        g.shape = d["shape"]
    n = g.n_v
    e = g.n_e
    print(n, e)
    np.random.seed(10)
    n = 75
    e = 4
    g = Graph(n, e)
    c = 1.55
    # m = no of observations
    m = int(c * ceil((e)*log(n, 2)))
    # m = int(c * ceil((e)*log(n, 2)**4))
    # Submatrix of of Hadamard Matrix that we are using
    print(n, e, m)
    print(g)
    # Create a mapping to know for which frequency each column of the p columns
    # of A belongs to
    measurement = {}
    for j in range(m):
        # Generate a random cut
        cut = [0]*n
        for i in range(n):
            cut[i] = np.random.randint(0, 2)
        cut = tuple(cut)
        cut_weight = g[cut]
        measurement[j] = (cut, cut_weight)
        #print("j=", j, "measurement[j]=", measurement[j])
    out = MatchingPursuit(measurement, n, m)
    y = {}
    for coor in out:
        y[tuple(getFreq(coor, n))] = -0.5
    print(y)
    y.pop(tuple([0]*n))
    g2 = Graph.create_from_FT(n, y)
    print(g == g2)
