from __future__ import print_function
import math
# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
import falconn
import timeit
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

# number of rows to read in the input csv files
my_nrows = 5000

#
# Init seed random generator
#
#np.random.seed()
#np.random.seed(13)
#np.random.seed(22409)
#np.random.seed(4057218)
#np.random.seed(5721840)
#np.random.seed(19680801)

# Build a virtual dataset
if not myCSV:
    print('========= BUILDING VIRTUAL DATASET =======')

N1 = 12
N  = N1*N1

x, y, z= np.meshgrid(np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32))
dataset = np.stack([x.flatten(), y.flatten(), z.flatten()], axis = -1)
dataset = dataset.astype(np.float32)
d = {}
for elem,ind in zip(dataset,np.arange(0,len(dataset))):
    d[tuple(elem)] = ind
dataset_copy = np.copy(dataset)


if not myCSV:
    print('========= READING DATASET ================')

som_dataset = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['dom', 'request', 'size','EcoIndex'],low_memory=False,nrows=my_nrows)
# normalize the 3rd column => divide by 1024 to convert it in KB
v = np.array([1,1,1024,1])
som_dataset = som_dataset / v
# Filter nul values
som_dataset = som_dataset[(som_dataset['dom'] > 0) & (som_dataset['request'] > 0) & (som_dataset['size'] > 0) ]
# Keep historical EcoIndex values
historical = som_dataset['EcoIndex']
#
# Convert the dataset to a numpy array
#
som_dataset = som_dataset.to_numpy()
som_dataset = som_dataset.astype(np.float32)

#print('Normalizing the dataset')
dataset /= np.linalg.norm(dataset, axis=1).reshape(-1,1)

#print('========= END READING ================')

# we build only 21 tables, increasing this quantity will improve the query time
# at a cost of slower preprocessing and larger memory footprint, feel free to
# play with this number
number_of_tables = 21

# It's important not to use doubles, unless they are strictly necessary.
# If your dataset consists of doubles, convert it to floats using `astype`.
# print(dataset.dtype)
#assert som_dataset.dtype == float32

params_cp = falconn.LSHConstructionParameters()
params_cp.dimension = 3#len(som_dataset[0])
params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
#params_cp.lsh_family = falconn.LSHFamily.Hyperplane
params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
#params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
params_cp.l = number_of_tables
# we set one rotation, since the data is dense enough,
# for sparse data set it to 2
params_cp.num_rotations = 1
params_cp.seed = 13 #5721840
# we want to use all the available threads to set up
params_cp.num_setup_threads = 0
params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
# we build 18-bit hashes so that each table has
# 2^18 bins; this is a good choice since 2^18 is of the same
# order of magnitude as the number of data points
falconn.compute_number_of_hash_functions(18, params_cp)

if not myCSV:
    print('========= CONSTRUCTING LSH TABLE =========')

table = falconn.LSHIndex(params_cp)
table.setup(dataset)

def ind(array, item):
        for idx, val in enumerate(array):
            #print(idx,val)
            if np.array_equal(val,item):
                return idx
    
import sys

average_RMSE = []
min_RMSE = 1000000000
max_RMSE = -1000000000

#############################################
for foo in range(1,2):

    y_actual = []
    y_predicted = []
 
    for x,known in zip(som_dataset,historical.to_numpy()):
        queries = [np.array([x[0],x[1],x[2]])]

        answers = []
        for query in queries:
            #print('Argmax:',np.dot(som_dataset, query).argmax())
            answers.append(np.dot(dataset, query).argmax())
        
        query_object = table.construct_query_object()

        def evaluate_number_of_probes(number_of_probes,queries,answers):
                query_object.set_num_probes(number_of_probes)
                score = 0
                for (i, query) in enumerate(queries):
                        #print('Type query',type(query),'Type query object:',type(query_object),'Type answers:',type(answers))
                        if answers[i] in query_object.get_candidates_with_duplicates(query):
                                score += 1
                return float(score) / len(queries)

        # find the smallest number of probes to achieve accuracy 0.9
        # using the binary search
        #print('Choosing number of probes')
        number_of_probes = number_of_tables
        
        while True:
            accuracy = evaluate_number_of_probes(number_of_probes,queries,answers)
            #print('{} -> {}'.format(number_of_probes, accuracy))
            if accuracy >= 0.9:
                break
            number_of_probes = number_of_probes * 2

        if number_of_probes > number_of_tables:
            left = number_of_probes // 2
            right = number_of_probes
            while right - left > 1:
                number_of_probes = (left + right) // 2
                accuracy = evaluate_number_of_probes(number_of_probes,queries,answers)
                #print('{} -> {}'.format(number_of_probes, accuracy))
                if accuracy >= 0.9:
                    right = number_of_probes
                else:
                    left = number_of_probes
            number_of_probes = right

        # final evaluation
        score = 0
        for (i, query) in enumerate(queries):
                #print('Query: ',queries_copy[i])
                if query_object.find_nearest_neighbor(query) == answers[i]:
                        score += 1
                        #print('found: ',query,' --> ',dataset_copy[answers[i]])
                #k_n = [query_object.find_nearest_neighbor(query)]
                k_n = query_object.find_k_nearest_neighbors(dataset_copy[answers[i]],N)
                centroid = []
                xx = 0
                yy = 0
                zz = 0
                for point in k_n:
                        XX = dataset_copy[point]
                        #print('Point:',XX,'at position:',d[tuple(XX)])
                        xx += XX[0] 
                        yy += XX[1] 
                        zz += XX[2] 
                centroid = [(xx/len(k_n)),(yy/len(k_n)),(zz/len(k_n))]

                #print('Centroid:',centroid,'Sum centroid:',sum(centroid),'Query:',query)
                predicted = 100 - 100*sum(centroid)/dataset_copy.max()/3 #sum(centroid)/3
                #print(sum(centroid),dataset_copy.max(),100*sum(centroid)/N/3)

                if not myCSV:
                    print('EcoIndex: {:.2f}'.format(100 - 100*sum(centroid)/dataset_copy.max()/3),'; Historical EcoIndex : {:.2f}'.format(known[0]))
                else:
                    print(x[0],';',x[1],';',x[2],'; {:.2f}'.format(known),'; {:.2f}'.format(100 - 100*sum(centroid)/dataset_copy.max()/3))
                y_actual.append(known)
                y_predicted.append(predicted)
                #print(predicted,known)
        
        MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
        RMSE = math.sqrt(MSE)
        average_RMSE.append(RMSE)
        min_RMSE = min(min_RMSE,RMSE)
        max_RMSE = max(max_RMSE,RMSE)
        #print(RMSE,min_RMSE,max_RMSE)
    
if not myCSV:
    from statistics import mean
    print("Average Root Mean Square Error:",mean(average_RMSE))
    print("Min Root Mean Square Error:",min_RMSE)
    print("Max Root Mean Square Error:",max_RMSE)

