from numpy import *;
from hadamard import fastwht;
from hadamardKernel import *;
import sys;

N  = 8;
x  = linspace(0,1,N);
x1 = x.copy();
x.shape = (N,1);
x1.shape = (1,N);
a = linalg.norm(x-x1);
print(a)
print(x-x1)


