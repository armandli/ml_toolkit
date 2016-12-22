#!/usr/bin/python2

import numpy as np
from numpy import pi #import the constant pi
from numpy import random #impor the random package

a = np.arange(15).reshape(3, 5) #create a array of 0..14 then reshape to matrix of 3 * 5
a.shape # return shape of a in row * col
a.ndim  # dimension of the array
a.dtype.name #type within the matrix
a.size  # number of cells within matrix
type(a) # returns ndarray

b = np.array([2,3,4]) # create a numpy array of 1 dim made of integers
c = np.array([[2,3,4], [5,6,7]]) # create 2 dim matrix
d = np.zeros((3, 4)) # tuple indiate the dimensions of row * col, this is a matrix of zeros
e = np.ones((2,3,4), dtype=float64) #create a matrix of 3 dim, with dtype float64
f = np.arange(10, 30, 5) # will generate a array of [10, 15, 20, 25, 30]

b**2      # square each cell
np.sin(a) # take the sin() of each cell in a
a < 5     # convert to a array of boolean answering the conditional
a * b     # cell by cell multiply
a.dot(b)  # matrix multiply, or np.dot(a, b)
random.random((2,3)) # generate random matrix in dimension 2,3
np.exp(b) # take the exponent per cell
b.sum(axis=0) # sum of each column, =1 means row
b.min(axis=0) # take the min of each column
n.sqrt(b) # take the square root
np.add(a, b) # cell add

#indexing and slicing
a[2] #one dim index 2
a[2:4] # slice from 2 to 3
a[:6:2] # take from 6th element to 1st steping at 2
a[::-1] # reverse the one dim array
#indexing and slicing 2d array
b[:,1]  # the second vertical column
b[1:2, :] # the second and third horizontal row
b[-1] # the last row

a.ravel() # flatten the array
a.T # transpose
a.resize(r, c) # resize the array, placing each element starting row by row

#stacking matrixes
a.vstack(a, b) #vertical stacking matrix a and b
a.hstack(a, b) #horizontal stacking matrix a and b

#hsplit and vsplit splits the matrix


