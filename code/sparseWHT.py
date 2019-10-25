import utils
import numpy as np
from math import ceil, log, isclose, floor
import hashing
from WHT import WHT
from randomGraph import Graph
from itertools import product
from utils import inp
from dataGraph import DataGraph
import pickle
from decision_tree import DecisionTree
import statistics


class SWHT(object):
    def __init__(self, n, K, C=1.0, ratio=1.5, coordBin=2, iter=2):
        # C bucket constant
        self.C = C
        # Size of Ground set
        self.n = n
        # Sparsity
        self.K = K
        # iterations for frequency detection
        self.iter = iter
        # try:
        #     self.degree = param["degree"]
        # except KeyError:
        #     self.degree = None
        # no bin coordinates are hashed to
        self.coordBin = coordBin
        # Bucket halving ratio
        self.ratio = ratio

    def takeMajorityVote(self, detectedFreq, rep, thresh=1):
        if(rep == 1):
            return detectedFreq[0]
        freqCount = {}
        freqValue = {}
        for j in range(rep):
            for freq in detectedFreq[j]:
                try:
                    freqCount[freq] += 1
                except KeyError:
                    freqCount[freq] = 1
                    freqValue[freq] = detectedFreq[j][freq]
        detectedFreq = {}
        for freq in freqCount:
            if freqCount[freq] >= thresh:
                detectedFreq[freq] = freqValue[freq]
        return detectedFreq

    def chooseFrequencies(self, estFreq, hash, rep=1):
        # freq = {}
        # print("Taking Majority Vote")
        # print(estFreq)
        # return estFreq[0]
        returnValue = {}
        for j in range(rep):
            delete = []
            for bucket in estFreq[j]:
                freq = estFreq[j][bucket][0]
                hashedFreq = hash.do_FreqHash(freq)
                if bucket != hashedFreq:
                    delete.append(bucket)
            for bucket in delete:
                del estFreq[j][bucket]
            for bucket in estFreq[j]:
                freq = estFreq[j][bucket][0]
                amplitude = estFreq[j][bucket][1]
                if np.sum(freq) == 2:
                    returnValue[bucket] = (freq, amplitude)
                if np.sum(freq) == 0 and np.sum(bucket) == 0:
                    returnValue[bucket] = (freq, amplitude)
                # elif np.sum(freq) == 0 and np.sum(bucket) != 0:
                #     print("Ha")
        # print(returnValue)
        return returnValue
        # for j in range(rep):
        #     for bucket in estFreq[j]:
        #         try:
        #             freq[bucket].append((tuple(estFreq[j][bucket][0]), estFreq[j][bucket][1]))
        #         except KeyError:
        #             freq[bucket] = []
        #             freq[bucket].append((tuple(estFreq[j][bucket][0]), estFreq[j][bucket][1]))
        # # print("freq=", freq)
        # estFreq = {}
        # for bucket in freq:
        #     # print("Bucket=", bucket)
        #     # print("freq[bucket]=", freq[bucket])
        #     try:
        #         mode = statistics.mode(freq[bucket])
        #     except:
        #         statistics.StatisticsError
        #         continue
        #     if freq[bucket].count(mode) >= thresh:
        #         estFreq[bucket] = [list(mode[0]), mode[1]]
        # # print("estFreq=", estFreq)
        # return estFreq

    def run(self, x):
        # B = no of bins we are hashing to
        # B = 48 * self.K
        B = int(self.K * self.C)
        # print("B=", B)
        b = int(ceil(log(B, 2)))
        # T = no. of iterations
        # T = int(min(floor(log(B, 2)) - 1, ceil(log(self.K, 2)) + 1))
        # T = ceil(log(self.K,4))
        T = int(floor(log(B, self.ratio)))-1
        # print(T)
        # T = int(min(floor(log(B, 1.6)) - 1, 10*ceil(log(self.K, 2)) + 1))
        # print(T)
        # est will hold as key frequencies and as value amplitudes
        est = {}
        detectedFreq = {}
        for i in range(T):
            # print("Iteration ", i)
            # print("B=", B, "b=", b)
            # Majority Vote
            for j in range(1):
                # Define a new hashing matrix A
                hash = hashing.Hashing(self.n, b)
                # hashedEstimate will hold as keys bin frequencies and as values
                # tuples where the first element is the freq hashed to that bin
                # and second element is the amplitude
                hashedEst = self.hashFrequencies(hash, est)
                # print("hashedEstimate", hashedEst)
                detectedFreq[j] = self.detectFreqLowDegree(x, hash, hashedEst, "graph", self.iter)
                # print(detectedFreq[j])
                # print("Detected Frequencies ", detectedFreq)
            # Do majority Vote
            detectedFreq = self.takeMajorityVote(detectedFreq, 1, 1)
            # print("After Majority Vote")
            # print(detectedFreq)
            #########################
            # x.statistic(detectedFreq)
            # bucketCollision = {}
            # for edge in x.graph:
            #     freq = np.zeros((self.n))
            #     freq[edge[0]] = 1
            #     freq[edge[1]] = 1
            #     freq = tuple(freq)
            #     print(edge, "hashed to ", hash.do_FreqHash(freq))
            #     try:
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            #     except KeyError:
            #         bucketCollision[hash.do_FreqHash(freq)] = []
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            # collisions = 0
            # for bucket in bucketCollision:
            #     if len(bucketCollision[bucket]) > 1:
            #         collisions += len(bucketCollision[bucket])
            #         print(bucketCollision[bucket])
            # print("collisions=", collisions)
            ##########################
            # Run iterative updates
            for freq in detectedFreq:
                if freq in est:
                    est[freq] = est[freq] + detectedFreq[freq]
                    if isclose(est[freq], 0.0, abs_tol=0.0001):
                        # print("deleting", freq)
                        del est[freq]

                else:
                    est[freq] = detectedFreq[freq]

            # Buckets sizes for hashing reduces by half for next iteration
            B = int(ceil(B/self.ratio))
            b = int(ceil(log(B, 2)))
        return est

    def hashFrequencies(self, hash, est):
        # This function hashes the current estimated frequencies
        # of the signal to the buckets
        hashedEstimate = {}
        for key in est:
            hashed_key = hash.do_FreqHash(key)
            if hashed_key not in hashedEstimate:
                # Initialize empty list
                hashedEstimate[hashed_key] = []
            hashedEstimate[hashed_key].append((key, est[key]))
        return hashedEstimate

    def detectFrequency(self, x, hash, hashedEst):
        # We need the WHT with shift 0 for reference
        a = np.zeros((self.n), dtype=np.int64)
        # print("Before Zero shift ", str(x.sampCplx))
        hashedSignal = hash.do_TimeHash(x, a)
        # print("After Zero shift ", str(x.sampCplx))
        # print("hashed_signal=", hashedSignal)
        ref_signal = WHT(hashedSignal)
        # print("reference signal")
        # print(hashedWHT["ref"])
        # This dictionary will hold the WHTs of the subsampled signals
        hashedWHT = {}
        # Subsample Signal
        for j in range(self.n):
            # set a = e_j
            # print("e=", j)
            a = np.zeros((self.n), dtype=np.int64)
            a[j][0] = 1
            print(a)
            hashedSignal = hash.do_TimeHash(x, a)
            hashedWHT[j] = WHT(hashedSignal)
            # print(hashedWHT[j].shape)

        # Detect Frequencies
        # Dictionary of detected frequencies
        detectedFreq = {}
        # i is the number of the bucket we are checking in the iterations below
        for i in range(hash.B):
            bucket = self.toBinaryTuple(i, hash.b)
            # Compute the values of the esitmated signal hashed to this bucket
            if bucket in hashedEst:
                for X in hashedEst[bucket]:
                    ref_signal[bucket] = ref_signal[bucket] - X[1]

            # Only continue if a frequency with non-zero amplitude is hashed to bucket j
            # print("cheching ref_signal", ref_signal[bucket])
            if isclose(ref_signal[bucket], 0.0, abs_tol=0.0001):
                # print("Entered if statement for ref_signal[bucket]=0")
                continue
            # Subtract the current hashed estimates signal from each
            # of the buckets
            for j in range(self.n):
                if bucket in hashedEst:
                    for X in hashedEst[bucket]:
                        if(X[0][j] == 0):
                            hashedWHT[j][bucket] = hashedWHT[j][bucket] - X[1]
                        else:
                            hashedWHT[j][bucket] = hashedWHT[j][bucket] + X[1]
            # freq is the frequecy preset in this bin
            freq = [0]*self.n
            for j in range(self.n):
                if(np.sign(hashedWHT[j][bucket]) == np.sign(ref_signal[bucket])):
                    freq[j] = 0
                else:
                    freq[j] = 1
            detectedFreq[tuple(freq)] = ref_signal[bucket]
        # print (ref_signal)
        return detectedFreq

    # This function is used to compute shifts in subsampling the signal
    # in the low degree case
    def computeShift(self, coord_map, bin, bit):
        # Iterates through all bins that have a coordinate mapped to them
        mask = np.zeros(self.n, dtype=int)
        try:
            for c in coord_map[bin]:
                mask[c] = 1
        except KeyError:
            print("Bin is empty in call to computeShift")
        if(bit == "ref"):
            a = np.ones(self.n, dtype=int)
        else:
            a = np.arange(self.n)
            a = np.floor(a/(2 ** bit))
            a = (1 - (-1) ** a)/2
            a = a.astype(int)
        shift = np.multiply(a, mask)
        return shift

    # This function computes the inner product of two 0-1 n-tuples
    def inp(a, b):
        # print("inp", size(a), size(b))
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) % 2

    def detectFreqLowDegree(self, x, hash, hashedEst, d="graph", iter=1):
        # print("In method detectFreqLowDegree")

        estFreqIter = {}
        for i in range(iter):
            a = np.zeros((self.n), dtype=np.int64)
            hashedSignal = hash.do_TimeHash(x, a)
            ref_signal = WHT(hashedSignal)
            # Run the iterative algorithm for guessing the frequencies#
            # B = no of bins for hashing coordinates
            # B = 16 * d
            # if B < 4:
            #    B = 4
            # Special Hack for Graphs
            if d == "graph":
                B = self.coordBin
            else:
                B = int(2 * d * self.C)
            # T = no. of iterations
            # T = int(min(floor(log(B, 2)) - 1, ceil(log(d, 2))+1))
            # T = int(floor(log(B,2)))
            T = 1
            # T = ceil(log(self.K,4))
            # This array will hold the freq and amplitude present in each bucket
            estFreq = {}
            for j in range(T):
                # print("In freqeuency detection iteration", j)
                # print("B=", B)
                detected = self.hashCoordAndDetect(x, hash, hashedEst, estFreq, B, ref_signal)
                for bucket in detected:
                    if bucket in estFreq:
                        estFreq[bucket][0] = (estFreq[bucket][0] + detected[bucket][0]) % 2
                    else:
                        estFreq[bucket] = detected[bucket]
                # print("We detected the following frequencies", detected)
                # print("Current Estimate is", estFreq)
                B = int(ceil(B/2))
            estFreqIter[i] = estFreq
        estFreq = self.chooseFrequencies(estFreqIter, hash, iter)
        returnValue = {}
        # Check if buckets contain the correct frequencies
        delete = []
        for bucket in estFreq:
            freq = estFreq[bucket][0]
            hashedFreq = hash.do_FreqHash(freq)
            if bucket != hashedFreq:
                delete.append(bucket)
        for bucket in delete:
            del estFreq[bucket]

        # Check if buckets contain the correct frequencies by
        # Extra evaluations
        for _ in range(0):
            a = [np.random.randint(0, 2) for _ in range(self.n)]
            a = np.array(a)
            hashedSignal = hash.do_TimeHash(x, a)
            test_signal = WHT(hashedSignal)
            delete = []
            for bucket in estFreq:
                if bucket in hashedEst:
                    for X in hashedEst[bucket]:
                        if inp(X[0], a) == 0:
                            test_signal[bucket] = test_signal[bucket] - X[1]
                        else:
                            test_signal[bucket] = test_signal[bucket] + X[1]
                if inp(estFreq[bucket][0], a) == 0:
                    sign = 1
                else:
                    sign = -1
                if not isclose(test_signal[bucket], estFreq[bucket][1]*sign):
                    delete.append(bucket)
            for bucket in delete:
                print("Hello", bucket, estFreq[bucket])
                del estFreq[bucket]
        # Special Hack for graphs
        if d == "graph":
            for value in estFreq.values():
                if np.sum(value[0]) == 2 or np.sum(value[0]) == 0:
                    returnValue[tuple(value[0])] = value[1]
                else:
                    print("Blah")
                    returnValue[tuple(value[0])] = value[1]
        else:
            for value in estFreq.values():
                returnValue[tuple(value[0])] = value[1]
        return returnValue
    # Take majority vote

    def hashCoordAndDetect(self, x, hash, hashedEst, estFreq, B, ref_signal):

        # Hash the coordinates
        coord_map = {}
        for j in range(self.n):
            bin = np.random.randint(0, B)
            if bin not in coord_map:
                coord_map[bin] = []
            coord_map[bin].append(j)
        # print("coord_map=", coord_map)
        # We need to run a binary search on each subset of the coordinates
        # This dictionary will hold the WHTs of the subsampled signals
        hashedWHT = {}
        # print(coord_map)
        for bin in coord_map:
            # Subsample Signal #
            # This coressponds to running a binary search on the coordinates
            # in the given bin
            bitRange = list(range((int(ceil(log(self.n, 2))))))
            bitRange.append("ref")
            # print("bitrange=", bitRange)
            for bit in bitRange:
                # bit refers to which bit of the location
                # of the single 1 this binary search will specify
                shift = self.computeShift(coord_map, bin, bit)
                # print("bit=", bit, "bin=", bin, "shift=", shift)
                key = tuple(np.squeeze(shift).tolist())
                # print("before shift =", key, "sampClpx=", str(x.sampCplx))
                hashedSignal = hash.do_TimeHash(x, shift)
                # print("after shift =", key, "sampClpx=", str(x.sampCplx))
                hashedWHT[(bin, bit)] = WHT(hashedSignal)
                # print(hashedWHT[j].shape)
                # bit = "ref" This will help us distinguish between case where
                # the freq constrained to the bin is all zeros or not

        # Detect Frequencies
        # Dictionary of residual frequencies in each bucket
        resFreq = {}
        # print("Current estimated Freq is", estFreq)
        # i is the number of the bucket we are checking in the iterations below

        for i in range(hash.B):
            debug = False
            bucket = self.toBinaryTuple(i, hash.b)
            if bucket == (1, 1, 0, 0):
                debug = False
            # Compute the values of the esitmated signal hashed to this bucket
            if bucket in hashedEst:
                if debug == True:
                    print("Hello", bucket, hashedEst[bucket], ref_signal[bucket])

                for X in hashedEst[bucket]:

                    ref_signal[bucket] = ref_signal[bucket] - X[1]

            # Only continue if a frequency with non-zero amplitude is hashed to bucket j
            # print("cheching ref_signal", ref_signal[bucket])
            if isclose(ref_signal[bucket], 0.0, abs_tol=0.0001):
                # print("Entered if statement for ref_signal[bucket]=0")
                if debug == True:
                    print("Didn't even continue")
                continue
            # Subtract the current hashed estimates signal from each
            # of the buckets
            for bin, bit in product(coord_map, bitRange):
                if bucket in hashedEst:
                    if debug == True:
                        print("Subtracting old signal")
                    for X in hashedEst[bucket]:
                        if(inp(X[0], self.computeShift(coord_map, bin, bit)) == 0):
                            hashedWHT[(bin, bit)][bucket] = hashedWHT[(bin, bit)][bucket] - X[1]
                        else:
                            hashedWHT[(bin, bit)][bucket] = hashedWHT[(bin, bit)][bucket] + X[1]
            # Easy check to see if bucket has more than two
            # frequencies hashed to it
            flag = True
            for bin in coord_map:
                for bit in bitRange:
                    if not isclose(abs(ref_signal[bucket]), abs(hashedWHT[(bin, bit)][bucket]), abs_tol=0.0001):
                        flag = False
                        break
            if flag == False:
                #     #     print("bye", bucket)
                if debug == True:
                    print("bye")
                continue

            if bucket in estFreq:
                eFreq = estFreq[bucket][0]
            else:
                eFreq = np.array([0]*self.n)
            # print("current estFreq for bucket", bucket, "is", eFreq)

            # resFreq is the residual frequency in this bucket
            # i.e the frequency in the bucket minus the previous estimated freq
            rFreq = np.array([0]*self.n)
            for bin in coord_map:
                if debug == True:
                    print("Bin", bin)
                # print("Computing Residual Frequency in bin", bin)
                # We need the esitmate frequency in this bucket if it exists
                # from a previous step

                # Run binary search for each subset of coordinates
                position = 0
                for bit in range(int(ceil(log(self.n, 2)))):
                    eSign = inp(eFreq, self.computeShift(coord_map, bin, bit))
                    if(np.sign(hashedWHT[(bin, bit)][bucket]) != np.sign(ref_signal[bucket])):
                        newSign = 1
                    else:
                        newSign = 0
                    if((eSign+newSign) % 2 == 1):
                        position = position + 2**bit
                if debug == True:
                    print("Position", position)
                # In case position = 0 , we need to check if it is the case
                # that the position of the single bit in zero position
                # or the frequncy constrain to those subset of bits has no
                # ones
                # print("position is", position)
                if(position not in coord_map[bin]):
                    # There are more than two ones in the freq constrained to
                    # the coordinates so the result is rubbish
                    continue
                if(position == 0):
                    eSign = inp(eFreq, self.computeShift(coord_map, bin, "ref"))
                    if(np.sign(hashedWHT[(bin, "ref")][bucket]) != np.sign(ref_signal[bucket])):
                        newSign = 1
                    else:
                        newSign = 0
                    if(newSign+eSign == 1):
                        rFreq[0] = 1

                else:
                    rFreq[position] = 1
                if debug == True:
                    print("rFreq", rFreq)
            resFreq[bucket] = [rFreq, ref_signal[bucket]]
            # print("Residual frequency for bucket", bucket, "is", rFreq)
        # print("Returning from hashCoordAndDetect with resFreq=", resFreq)
        return resFreq

    def toBinaryTuple(self, i, b):
        # Converts integer i into an b-tuple of 0,1s
        a = list(bin(i)[2:].zfill(b))
        a = tuple([int(x) for x in a])
        return a


if __name__ == "__main__":
    np.random.seed(4)
    params = utils.load_config()
    n = 30
    e = 15
    swht = SWHT(n, e, 1.1, 1.4, 3, 1)
    # x = np.array([0, 2, 2, 0, 5, 7, 7, 5, 5, 7, 7, 5, 0, 2, 2, 0])
    # x = x.reshape(tuple([2]*4))
    # print(x)
    # print(x[(0, 0, 0, 0)])
    # print(x[(0, 0, 0, 1)])
    # print(x[(0, 0, 1, 0)])
    # print(x[(0, 0, 1, 1)])
    # print(x[(0, 1, 0, 0)])
    # print(x[(0, 1, 0, 1)])
    # print(x[(0, 1, 1, 0)])
    # print(x[(0, 1, 1, 1)])
    # print(x[(1, 0, 0, 0)])
    # print(x[(1, 0, 0, 1)])
    # print(x[(1, 0, 1, 0)])
    # print(x[(1, 0, 1, 1)])
    # print(x[(1, 1, 0, 0)])
    # print(x[(1, 1, 0, 1)])
    # print(x[(1, 1, 1, 0)])
    # print(x[(1, 1, 1, 1)])
    # y = swht.run(x)
    # print(y)

    # g = DataGraph(0, {})
    # with open('GraphData/1.pkl', 'rb') as f:
    #     d = pickle.load(f)
    #     g.graph = d["graph"]
    #     g.n_v = d["n_v"]
    #     g.n_e = d["n_e"]
    #     g.shape = d["shape"]
    #     g.sampCplx = 0

    # g = DecisionTree(swht.n, swht.K, swht.degree)
    g = Graph(swht.n, swht.K)
    print(g)
    p = 0
    for i in range(1):
        # print(i)
        y = swht.run(g)
        try:
            g2 = Graph.create_from_FT(swht.n, y)
        except AssertionError:
            continue
        if g == g2:
            p = p+1
        g.cache = {}
    print(p)

    # y = swht.run(g)
    # print(y, "\n", g)
    # y.pop(tuple([0]*swht.n))
    # g2 = Graph.create_from_FT(swht.n, y)
    # print(g == g2)
    print("SamplingComplexity =", g.sampCplx)
    # bit = 2
    # j = np.arange(20)
    # j = np.floor(j/(2 ** bit))
    # a = (1 - (-1) ** j)/2
    # a = a.astype(int)
    # print(a)
    # print(a.shape)
    # mask = np.zeros(20, dtype=int)
    # mask[4:6] = 1
    # mask[0:2] = 1
    # mask[14:16] = 1
    # print(mask.shape)
    # print(mask)
    # r = np.multiply(a, mask).reshape(20, 1)
    # print(r, r.shape)
    # bitIndexRange = list(range((int(ceil(log(100, 2))))))
    # bitIndexRange.append("ref")
    # print(bitIndexRange)
    # for j in bitIndexRange:
    #     print(j)
