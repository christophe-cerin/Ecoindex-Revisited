#!/usr/bin/env python3
"""
Example file to illustrate the EcoIndex Computation through
the collinearity method

Usage to compute the EcoIndex for the request (1*3, 1*3, 1*1) with
a virtual space of size 9*9*9

$ python3  collinearity.py 1 1 1 3 3 1 9
Arguments count: 8
Argument      0: collinearity.py
Argument      1: 1
Argument      2: 1
Argument      3: 1
Argument      4: 3
Argument      5: 3
Argument      6: 1
Argument      7: 9
Query          : [3, 3, 1]
Normalizing the dataset of length: 729
Dataset normalized
Final centroid: [2.6296296296296298, 2.6049382716049383, 0.8395061728395061]
EcoIndex: 97.50
Query time: 0.015776100000948645
We used a 3-d virtual space of 729 random 3d points
"""

from __future__ import print_function
import numpy as np
import timeit
import math
import sys
from operator import itemgetter
import itertools
import random
from scipy.spatial import distance

__author__ = "Christophe Cerin"
__copyright__ = "Copyright 2022"
__credits__ = ["Christophe Cerin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christophe Cerin"
__email__ = "christophe.cerin@univ-paris13.fr"
__status__ = "Experimental"

# Function to check if two given
# vectors are collinear or not
def ComputeCollinearity(x1, y1, z1, x2, y2, z2):
     
    # Store the first and second vectors
    A = [x1, y1, z1]
    B = [x2, y2, z2]

    cross_P = np.cross(A,B)

    #print(cross_P)

    return cross_P

    # Check if their cross product
    # is a NULL Vector or not
    if (cross_P[0] == 0 and
        cross_P[1] == 0 and
        cross_P[2] == 0):
        print("Yes")
    else:
        print("No")
 
#
# Build the virtual 3-d space (method 1)
#
def make_cubes_one(dataset,dim,res):
    #print('Iteration number:',int(len(dataset)/dim),'Dim:',dim)
    num = 0
    for i in range(0,int(len(dataset)/dim),1):
        XX = dataset[i*dim:i*dim+dim]
        #print('---',XX)
        pos = 0; inc = int(math.sqrt(len(XX)))
        for j in XX:
            #print('Point:',j[0],' ',j[1],' ',j[2])
            x = 0.0 ; y = 0.0 ; z = 0.0
            #print('-->',len(XX[pos:pos+inc]),' - ',len(XX))
            for k in XX[pos:pos+inc]:
                x += k[0]
                y += k[1]
                z += k[2]
            centroid = [x/len(XX[pos:pos+inc]),y/len(XX[pos:pos+inc]),z/len(XX[pos:pos+inc])]
            pos += inc
            print('Point',i,' ',centroid)
            res[num] = centroid
            num += 1

#
# build the virtual 3-d space (method 2)
#
def make_cubes_random(dataset,dim,res):
    #xaxis = []; yaxis = []; zaxis = [];
    mycube=dim*dim*dim
    num = 0
    for ind in range(0,len(dataset),mycube):
        XX = np.copy(dataset[ind:ind+mycube])
        centroid = random.choice(random.choices(XX, weights=map(len, XX)))
        #print('Random centroid:',centroid)
        res[num] = centroid
        num += 1

#
# build the virtual 3-d space (method 3)
#
def make_cubes(dataset,dim,res):
    #xaxis = []; yaxis = []; zaxis = [];
    mycube=dim*dim*dim
    num = 0
    for ind in range(0,len(dataset),mycube):
        XX = np.copy(dataset[ind:ind+mycube])
        x = np.sum(a=XX,axis=0)
        centroid = [x[0]/mycube,x[1]/mycube,x[2]/mycube,]
        res[num] = centroid
        num += 1
        #xaxis.append([x[0]/mycube]);
        #yaxis.append([x[1]/mycube]);
        #zaxis.append([x[2]/mycube]);
        #print(XX)
        #print(ind,' -> ',x)
        #print('Centroid:',centroid)

    
if __name__ == '__main__':

    if len(sys.argv) != 8:
         print("Bad number of argument. Require an URL as parameter!")
         print('Usage: python3 collinearity.py dom request size weight_dom weight_request weight_size')
         print('All parameters should be integers!')
         exit()

    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
         print(f"Argument {i:>6}: {arg}")
         if i == 1:
             dom = int(arg)
         if i == 2:
             request = int(arg)
         if i == 3:
             size = int(arg)
         if i == 4:
             weight_dom = int(arg)
         if i == 5:
             weight_request = int(arg)
         if i == 6:
             weight_size = int(arg)
         if i == 7:
             N1 = int(arg)

    #
    # build th request we are looking for the ecoindex
    #
    query = [ dom * weight_dom, request * weight_request, size * weight_size]
    query_norm = query
    query_norm /= np.linalg.norm(query_norm, axis=0).reshape(-1, 1) 
    print('Query          :',query)
    #print('Query          :',query_norm[0])
             
    N  = N1*N1
    x, y, z = np.meshgrid(np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32))
    dataset = np.stack([x.flatten(), y.flatten(), z.flatten()], axis = -1)
    #print(dataset)
    #print('Nb elements of our dataset:',len(dataset))

    #
    # Make cubes. We replace N points by a single point i.e,. the centroid
    #
    #dataset_bak = np.empty([int(len(dataset)/N),3],dtype=np.float32)
    dataset_bak = np.full((int(len(dataset)/(N1*N1*N1)),3),np.float32(0.0))
    make_cubes_random(dataset,N1,dataset_bak)
    #for ind, i in enumerate(dataset_bak):
    #     print(ind,':',i)

    #d = {}
    #for elem,ind in zip(dataset,np.arange(0,len(dataset_bak))):
    #    d[tuple(elem)] = ind
    dataset_copy = np.copy(dataset_bak)
    #print(dataset)

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    # print(dataset.dtype)
    assert dataset.dtype == np.float32

    # Normalize all the lenghts, since we care about the cosine similarity.
    print('Normalizing the dataset of length:',len(dataset_bak))
    dataset_bak /= np.linalg.norm(dataset_bak, axis=1).reshape(-1, 1)
    print('Dataset normalized')
    #print(dataset_bak)
    #print('Done')

    d = {}
    for (elem, value) in zip(dataset_bak, dataset_copy):
        d[tuple(elem)] = value
    #for key in d:
    #    print(key, '->', d[key])
    
    #def ind(array, item):
    #    for idx, val in enumerate(array):
    #        #print(idx,val)
    #        if np.array_equal(val,item):
    #            return idx

    t1 = timeit.default_timer()

    res = []
    dd = {}
    for i in dataset_bak:
        res1 = ComputeCollinearity(query_norm[0][0], query_norm[0][1], query_norm[0][2],i[0], i[1], i[2])
        dd[tuple(res1)] = i
        res = res + [res1]
    #print(res)
    #print(dd)

    # sort the list of points
    sorted_points = sorted(res, key=itemgetter(0,1,2))
    #for i in sorted_points:
    #    print(i)

    #
    # Compute the distances
    #
    my_dist = distance.cdist(sorted_points, query_norm, 'euclidean')
    #print('My distance:',my_dist)
    #print('Index of min:',np.where(my_dist == my_dist.min()))
    test_list = list(itertools.chain.from_iterable(my_dist))
    K = 3
    idx = sorted(range(len(test_list)), key = lambda sub: test_list[sub])[:K]
    #print('Index of',K,'minimal values:',idx)
    x = 0.0 ; y = 0.0 ; z = 0.0
    for i in idx:
        pool = d[tuple(dd[tuple(sorted_points[i])])]
        x += pool[0]
        y += pool[1]
        z += pool[2]
    centroid = [x/N,y/N,z/N]
    print('Final centroid:',centroid)
    print('EcoIndex: {:.2f}'.format(100 - 100*sum(centroid)/dataset_copy.max()/3))
    
    t2 = timeit.default_timer()

    print('Query time: {}'.format((t2 - t1)))
    print('We used a 3-d virtual space of',len(res),'random 3d points')
    #print(dataset_copy)
