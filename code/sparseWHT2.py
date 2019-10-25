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


class SWHT(object):
    def __init__(self, n, K, C=1.0):
        # C bucket constant
        self.C = C
        # Size of Ground set
        self.n = n
        # Sparsity
        self.K = K

        # try:
        #     self.degree = param["degree"]
        # except KeyError:
        #     self.degree = None

    def run(self, x):
        # B = no of bins we are hashing to
        # B = 48 * self.K
        B = int(2.2 * self.K * self.C)
        #print("B=", B)
        b = int(ceil(log(B, 2)))
        # T = no. of iterations
        T = int(min(floor(log(B, 1.32)) - 1, 10*ceil(log(self.K, 2)) + 1))
        # T = ceil(log(self.K,4))

        # est will hold as key frequencies and as value amplitudes
        est = {}
        for i in range(T):
            # Define a new hashing matrix A
            # print("Iteration ", i)
            # print("B=", B, "b=", b)
            hash = hashing.Hashing(self.n, b)
            # hashedEstimate will hold as keys bin frequencies and as values
            # tuples where the first element is the freq hashed to that bin
            # and second element is the amplitude
            hashedEst = self.hashFrequencies(hash, est)
            # print("hashedEstimate", hashedEst)
            detectedFreq = self.detectFreqLowDegree(x, hash, hashedEst, 2)
            # print("Detected Frequencies ", detectedFreq)
            # Run iterative updates
            for freq in detectedFreq:
                if freq in est:
                    est[freq] = est[freq] + detectedFreq[freq]
                    #print(freq, detectedFreq[freq])
                    if isclose(est[freq], 0.0, abs_tol=0.0001):
                        # print("deleting", freq)
                        del est[freq]

                else:
                    est[freq] = detectedFreq[freq]
                    #print(freq, est[freq])

            # Buckets sizes for hashing reduces by half for next iteration
            B = int(ceil(B/1.32))
            b = int(ceil(log(B, 2)))
        return est

    def hashFrequencies(self, hash, est):
        # This function hashes the current estimated frequencies
        # of the signal to the buckets
        hashedEstimate = {}
        for key in est:
            hashed_key = hash.do_FreqHash(key)
            if hashed_key not in hashedEstimate:
                # Initialize empty list
                hashedEstimate[hashed_key] = []
            hashedEstimate[hashed_key].append((key, est[key]))
        return hashedEstimate

    def detectFrequency(self, x, hash, hashedEst):
        # We need the WHT with shift 0 for reference
        a = np.zeros((self.n, 1), dtype=np.int64)
        hashedSignal = hash.do_TimeHash(x, a)
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
            a = np.zeros((self.n, 1), dtype=np.int64)
            a[j][0] = 1
            # print(a)
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
        shift = np.multiply(a, mask).reshape(self.n, 1)
        return shift

    # This function computes the inner product of two 0-1 n-tuples
    def inp(a, b):
        # print("inp", size(a), size(b))
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) % 2

    def detectFreqLowDegree(self, x, hash, hashedEst, d=2, iter=1):
        # print("In method detectFreqLowDegree")
        out = {}
        for i in range(iter):
            # Run the iterative algorithm for guessing the frequencies#
            # B = no of bins for hashing coordinates
            # B = 16 * d
            B = int(2 * d * self.C)
            if B < 5:
                B = 3
            # T = no. of iterations
            T = 1
            #int(min(ceil(log(B, 2)), ceil(log(d, 2))+1))
            # T = ceil(log(self.K,4))
            # This array will hold the freq and amplitude present in each bucket
            estFreq = {}
            for j in range(T):
                # print("In freqeuency detection iteration", j)
                # print("B=", B)
                detected = self.hashCoordAndDetect(x, hash, hashedEst, estFreq,  d, B)
                for bucket in detected:
                    if bucket in estFreq:
                        estFreq[bucket][0] = (estFreq[bucket][0] + detected[bucket][0]) % 2
                    else:
                        estFreq[bucket] = detected[bucket]
                # print("We detected the following frequencies", detected)
                # print("Current Estimate is", estFreq)
                B = int(ceil(B/2))
            out[i] = estFreq
        # print(out[0])
        # TODO: Change this to a majority vote taken over each of the buckets
        returnValue = {}
        for value in out[0].values():
            temp_freq = np.array(value[0])
            if np.sum(temp_freq) == 1 or np.sum(temp_freq) > 2:
                continue
            returnValue[tuple(value[0])] = value[1]

        return returnValue
    # Take majority vote

    def hashCoordAndDetect(self, x, hash, hashedEst, estFreq, d, B):
        # We need the WHT with shift 0 for reference
        a = np.zeros((self.n, 1), dtype=np.int64)
        hashedSignal = hash.do_TimeHash(x, a)
        # print("hashed_signal=", hashedSignal)
        ref_signal = WHT(hashedSignal)
        # print("reference signal")
        # print(hashedWHT["ref"])

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
        for bin in coord_map:
            # Subsample Signal #
            # This coressponds to running a binary search on the coordinates
            # in the given bin
            bitRange = list(range((int(ceil(log(self.n, 2))))))
            bitRange.append("ref")
            for bit in bitRange:
                # bit refers to which bit of the location
                # of the single 1 this binary search will specify
                shift = self.computeShift(coord_map, bin, bit)
                # print("bit=", bit, "bin=", bin, "shift=", shift)
                hashedSignal = hash.do_TimeHash(x, shift)
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
            for bin, bit in product(coord_map, bitRange):
                if bucket in hashedEst:
                    for X in hashedEst[bucket]:
                        if(inp(X[0], self.computeShift(coord_map, bin, bit)) == 0):
                            hashedWHT[(bin, bit)][bucket] = hashedWHT[(bin, bit)][bucket] - X[1]
                        else:
                            hashedWHT[(bin, bit)][bucket] = hashedWHT[(bin, bit)][bucket] + X[1]

            if bucket in estFreq:
                eFreq = estFreq[bucket][0]
            else:
                eFreq = np.array([0]*self.n)
            # print("current estFreq for bucket", bucket, "is", eFreq)

            # resFreq is the residual frequency in this bucket
            # i.e the frequency in the bucket minus the previous estimated freq
            rFreq = np.array([0]*self.n)
            for bin in coord_map:
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
                # In case position = 0 , we need to check if it is the case
                # that the position of the single bit in zero position
                # or the frequncy constrain to those subset of bits has no
                # ones
                # print("position is", position)
                if(position not in coord_map[bin]):
                    # There are more than two ones in the freq constrained to
                    # the coordinates so the result is rubbish
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
    # np.random.seed(13)
    # params = utils.load_config()
    n = 30
    e = 30
    swht = SWHT(n, e, 0.5)

    # g = DataGraph(0, {})
    # with open('GraphData/1.pkl', 'rb') as f:
    #     d = pickle.load(f)
    #     g.graph = d["graph"]
    #     g.n_v = d["n_v"]
    #     g.n_e = d["n_e"]
    #     g.shape = d["shape"]
    #     g.sampCplx = 0

    g = Graph(swht.n, swht.K)
    # g = DecisionTree(swht.n, swht.K, swht.degree)

    print(g)
    p = 0
    for i in range(100):
        y = swht.run(g)
        g2 = Graph.create_from_FT(swht.n, y)
        if g == g2:
            p = p+1
        g.cache = {}
    print(p)

    #y = swht.run(g)
    #print(y, "\n", g)
    # y.pop(tuple([0]*swht.n))
    #g2 = Graph.create_from_FT(swht.n, y)
    #print(g == g2)
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
