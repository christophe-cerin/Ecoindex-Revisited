from __future__ import print_function
import math
# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
import falconn
import timeit
import random            
from operator import itemgetter
import itertools
from scipy.spatial import distance
# Import libraries for plotting results
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Christophe Cerin"
__copyright__ = "Copyright 2023"
__credits__ = ["Christophe Cerin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christophe Cerin"
__email__ = "christophe.cerin@univ-paris13.fr"
__status__ = "Experimental"

# Boolean indicating if we generate a CSV format or not.
# In this last case we print the RMSE between the historical
# EcoIndex and the one computed with the 'colinearity method'.
myCSV = True

# Nomber of lines to read in the input csv files
my_nrows = 15000

#
# Init seed random generator
#
#np.random.seed()
#np.random.seed(13)
#np.random.seed(22409)
#np.random.seed(4057218)
#np.random.seed(5721840)
#np.random.seed(19680801)

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
            #print('Point',i,' ',centroid)
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

if not myCSV:        
    print('========= READING DATASET ================')

som_dataset = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['dom', 'request', 'size', 'EcoIndex'],low_memory=False,nrows=my_nrows)
# normalize the 3rd column => divide by 1024 to convert it in KB
v = np.array([1,1,1024,1])
som_dataset = som_dataset / v
# Filter nul values
som_dataset = som_dataset[(som_dataset['dom'] > 0) & (som_dataset['request'] > 0) & (som_dataset['size'] > 0) ]

#
# Convert the dataset to a numpy array
#
som_dataset = som_dataset.to_numpy()
som_dataset = som_dataset.astype(np.float32)

import sys

t1 = timeit.default_timer()

average_RMSE = []
min_RMSE = 1000000000
max_RMSE = -1000000000

# Size of the 3-d virtual space
N1 = 15
N  = N1*N1
x, y, z = np.meshgrid(np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32))
dataset = np.stack([x.flatten(), y.flatten(), z.flatten()], axis = -1)

#
# Make cubes. We replace N points by a single point i.e,. the centroid
#
#dataset_bak = np.empty([int(len(dataset)/N),3],dtype=np.float32)
dataset_bak = np.full((int(len(dataset)/(N1*N1*N1)),3),np.float32(0.0))
make_cubes_random(dataset,N1,dataset_bak)
dataset_copy = np.copy(dataset_bak)

# It's important not to use doubles, unless they are strictly necessary.
# If your dataset consists of doubles, convert it to floats using `astype`.
# print(dataset.dtype)
assert dataset.dtype == np.float32

# Normalize all the lenghts, since we care about the cosine similarity.
if not myCSV:
    print('Normalizing the dataset of length:',len(dataset_bak))
dataset_bak /= np.linalg.norm(dataset_bak, axis=1).reshape(-1, 1)
if not myCSV:
    print('Dataset normalized')
#print(dataset_bak)

d = {}
for (elem, value) in zip(dataset_bak, dataset_copy):
    d[tuple(elem)] = value

for foo in range(1,2):

    y_actual = []
    y_predicted = []
 
    for my_query in zip(som_dataset):

        #
        # build the request we are looking for the ecoindex
        #
        known = my_query[0][3]
        dom = my_query[0][0] ; weight_dom = 1; request = my_query[0][1] ; weight_request = 1; size = my_query[0][2] ; weight_size = 1
        if dom == 0 and request == 0 and size == 0:
            continue
        query = [ dom * weight_dom, request * weight_request, size * weight_size]
        query_norm = query
        query_norm /= np.linalg.norm(query_norm, axis=0).reshape(-1, 1) 
        #print('Query          :',query)

        res = []
        dd = {}
        for i in dataset_bak:
            res1 = ComputeCollinearity(query_norm[0][0], query_norm[0][1], query_norm[0][2],i[0], i[1], i[2])
            dd[tuple(res1)] = i
            res = res + [res1]

        # sort the list of points
        sorted_points = sorted(res, key=itemgetter(0,1,2))

        #
        # Compute the distances
        #
        my_dist = distance.cdist(sorted_points, query_norm, 'euclidean')
        #print('My distance:',my_dist)
        #print('Index of min:',np.where(my_dist == my_dist.min()))
        test_list = list(itertools.chain.from_iterable(my_dist))
        # We keep only K 'candidates' in the sorted list
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
        #print('Final centroid:',centroid,'Query:',query)
        predicted = 100 - 100*sum(centroid)/dataset_copy.max()/3
        if myCSV:
            print(dom * weight_dom,';', request * weight_request,';', size * weight_size,'; {:.2f}'.format(known),'; {:.2f}'.format(predicted))
        #else:
        #    print('Predicted EcoIndex: {:.2f}'.format(predicted),'; Historical EcoIndex:{:.2f}'.format(known))

        #print('We used a 3-d virtual space of',len(res),'random 3d points')

        y_actual.append(known)
        y_predicted.append(predicted)
        #print(predicted,known)
        
        MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
        RMSE = math.sqrt(MSE)
        average_RMSE.append(RMSE)
        min_RMSE = min(min_RMSE,RMSE)

t2 = timeit.default_timer()

if not myCSV:
    from statistics import mean
    print("Average Root Mean Square Error:",mean(average_RMSE))
    print("Min Root Mean Square Error:",min_RMSE)
    print("Max Root Mean Square Error:",max_RMSE)
    print('Compute time: {} per entry of the dataset'.format((t2 - t1) / my_nrows))
    print('Total completion time: {}'.format((t2 - t1)))
