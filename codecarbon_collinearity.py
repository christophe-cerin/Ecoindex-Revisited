#!/usr/bin/env python3
"""
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import timeit
import math
import sys
from operator import itemgetter
import itertools
import random
from scipy.spatial import distance

# codecarbon
from codecarbon import EmissionsTracker

__author__ = "Christophe Cerin"
__copyright__ = "Copyright 2023"
__credits__ = ["Christophe Cerin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christophe Cerin"
__email__ = "christophe.cerin@univ-paris13.fr"
__status__ = "Experimental"

# Function to check if two given
# vectors are collinear or not
def compute_collinearity(x1, y1, z1, x2, y2, z2):
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

    if len(sys.argv) != 1:
         print("Bad number of argument. No parameter!")
         exit()

    # dimension of the virtual 3-d space
    N1 = 11
    weight_dom = 3
    weight_request = 1
    weight_size = 2

    # Number of rows to read in the dataset
    my_nrows = 100000
        
    input_dataset = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['dom', 'request', 'size'],low_memory=False,nrows=my_nrows)
    # normalize the 3rd column => divide by 1024 to convert it in KB
    v = np.array([1,1,1024])
    input_dataset = input_dataset / v
    # Filter nul values
    input_dataset = input_dataset[(input_dataset['dom'] > 0) & (input_dataset['request'] > 0) & (input_dataset['size'] > 0) ]
    #
    # Convert the dataset to a numpy array
    #
    input_dataset = input_dataset.to_numpy()
    input_dataset = input_dataset.astype(np.float32)

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
    #print('Normalizing the dataset of length:',len(dataset_bak))
    dataset_bak /= np.linalg.norm(dataset_bak, axis=1).reshape(-1, 1)
    #print('Dataset normalized')
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

    tracker = EmissionsTracker()
    tracker.start()

    for dom, request, size in input_dataset:
    
        #
        # build the request we are looking for the eco_index
        #
        query = [ dom * weight_dom, request * weight_request, size * weight_size]
        query_norm = query
        query_norm /= np.linalg.norm(query_norm, axis=0).reshape(-1, 1) 
        #print('Query          :',query)
        #print('Query          :',query_norm[0])
             
        t1 = timeit.default_timer()

        res = []
        dd = {}
        for i in dataset_bak:
            res1 = compute_collinearity(query_norm[0][0], query_norm[0][1], query_norm[0][2],i[0], i[1], i[2])
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
        centroid = [x/K,y/K,z/K]
        #print('Final centroid:',centroid)
        #print('eco_index: {:.2f}'.format(100 - 100*sum(centroid)/dataset_copy.max()/3))
    
        t2 = timeit.default_timer()

        #print('Query time: {}'.format((t2 - t1)))
        #print('We used a 3-d virtual space of',len(res),'random 3d points')
        #print(dataset_copy)

    # stop codecarbon
    emissions: float = tracker.stop()
                
    print(f"Emissions: {emissions} kg for {my_nrows} URLs")
