#!/usr/bin/env python3
"""
Example file to illustrate the EcoIndex Computation through
the lsh method. Here we use the Python Falconn package for
lsh/k-NN computation

Usage:

$ python3 lsh.py
Normalizing the dataset
Done
Generating queries
Queries:  [array([35., 35., 29.], dtype=float32), array([17., 18., 39.], dtype=float32)]
Done
Solving queries using linear scan
Done
Linear scan time: 0.00016560000221943483 per query
Constructing the LSH table
Done
Construction time: 0.16697279999789316
Choosing number of probes
21 -> 1.0
Done
21 probes
found:  [0.6101043  0.6101043  0.50551504]  -->  [35. 35. 29.]
Centroid of the k nearest neighbors: [34.02040816326531, 34.02040816326531, 28.183673469387756]
EcoIndex: 34.54
found:  [0.3680033  0.38965055 0.8442429 ]  -->  [17. 18. 39.]
Centroid of the k nearest neighbors: [16.755102040816325, 17.816326530612244, 38.48979591836735]
EcoIndex: 50.30
Query time: 0.017494949999672826
Precision: 1.0
We considered a space of 117649 3d points
"""

from __future__ import print_function
import numpy as np
import falconn
import timeit
import math

__author__ = "Christophe Cerin"
__copyright__ = "Copyright 2022"
__credits__ = ["Christophe Cerin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christophe Cerin"
__email__ = "christophe.cerin@univ-paris13.fr"
__status__ = "Experimental"


N1 = 15
N  = N1*N1
x, y, z= np.meshgrid(np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32), np.arange(1, N+1,dtype=np.float32))
dataset = np.stack([x.flatten(), y.flatten(), z.flatten()], axis = -1)
d = {}
for elem,ind in zip(dataset,np.arange(0,len(dataset))):
    d[tuple(elem)] = ind
dataset_copy = np.copy(dataset)
#dataset=cube.reshape(N*N,N,3)
#print(dataset)

if __name__ == '__main__':

    number_of_queries = 2
    # we build only 21 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number
    number_of_tables = 21

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    # print(dataset.dtype)
    assert dataset.dtype == np.float32

    # Normalize all the lenghts, since we care about the cosine similarity.
    print('Normalizing the dataset')
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    #print('dataset normalized: ',dataset)
    print('Done')

    # Choose random data points to be queries.
    print('Generating queries')
    #np.random.seed(4057218)
    #np.random.shuffle(dataset)
    #queries = dataset[len(dataset) - number_of_queries:]
    queries = dataset[len(dataset) - number_of_queries:]
    #dataset = dataset[:len(dataset) - number_of_queries]
    queries = []
    queries_copy = []
    for i in range(number_of_queries):
        m = np.random.randint(len(dataset))
        #jj = np.array([(i+1)*np.float32(10),(i+1)*np.float32(10),(i+1)*np.float32(10)],dtype=np.float32)
        #queries.append(jj)
        queries.append(dataset[m])
        #queries_copy.append(jj)
        queries_copy.append(dataset_copy[m])
    print('Queries: ',queries_copy)
    print('Done')
    

    # Perform linear scan using NumPy to get answers to the queries.
    print('Solving queries using linear scan')
    t1 = timeit.default_timer()
    answers = []
    for query in queries:
        #print(np.dot(dataset, query).argmax())
        answers.append(np.dot(dataset, query).argmax())
    #print('answers:',answers)
    t2 = timeit.default_timer()
    print('Done')
    print('Linear scan time: {} per query'.format((t2 - t1) / float(
        len(queries))))

    # Center the dataset and the queries: this improves the performance of LSH quite a bit.
    #print('Centering the dataset and queries')
    #center = np.mean(dataset, axis=0)
    #dataset -= center
    #queries -= center
    #print('Done')

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
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

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2 = timeit.default_timer()
    print('Done')
    #print(dir(table))
    #print(dir(table._table))
    print('Construction time: {}'.format(t2 - t1))

    query_object = table.construct_query_object()

    # find the smallest number of probes to achieve accuracy 0.9
    # using the binary search
    print('Choosing number of probes')
    number_of_probes = number_of_tables

    def evaluate_number_of_probes(number_of_probes):
        query_object.set_num_probes(number_of_probes)
        score = 0
        for (i, query) in enumerate(queries):
            if answers[i] in query_object.get_candidates_with_duplicates(
                    query):
                score += 1
        return float(score) / len(queries)

    while True:
        accuracy = evaluate_number_of_probes(number_of_probes)
        print('{} -> {}'.format(number_of_probes, accuracy))
        if accuracy >= 0.9:
            break
        number_of_probes = number_of_probes * 2
    if number_of_probes > number_of_tables:
        left = number_of_probes // 2
        right = number_of_probes
        while right - left > 1:
            number_of_probes = (left + right) // 2
            accuracy = evaluate_number_of_probes(number_of_probes)
            print('{} -> {}'.format(number_of_probes, accuracy))
            if accuracy >= 0.9:
                right = number_of_probes
            else:
                left = number_of_probes
        number_of_probes = right
    print('Done')
    print('{} probes'.format(number_of_probes))

    def ind(array, item):
        for idx, val in enumerate(array):
            #print(idx,val)
            if np.array_equal(val,item):
                return idx
    
    # final evaluation
    t1 = timeit.default_timer()
    score = 0
    for (i, query) in enumerate(queries):
        #print('Query: ',queries_copy[i])
        if query_object.find_nearest_neighbor(query) == answers[i]:
            score += 1
            print('found: ',query,' --> ',dataset_copy[answers[i]])
        #k_n = [query_object.find_nearest_neighbor(query)]
        k_n = query_object.find_k_nearest_neighbors(query,N)
        #k_n = query_object.find_near_neighbors(query,0.0)
        #print('k neareast: ',k_n)
        centroid = []
        x = 0
        y = 0
        z = 0
        for point in k_n:
            XX = dataset_copy[point]
            #print('Point:',XX,'at position:',d[tuple(XX)])
            x += XX[0]
            y += XX[1]
            z += XX[2]
        centroid = [x/len(k_n),y/len(k_n),z/len(k_n)]
        print('Centroid of the k nearest neighbors:',centroid)
        print('EcoIndex: {:.2f}'.format(100 - 100*sum(centroid)/dataset_copy.max()/3))
            
    t2 = timeit.default_timer()

    print('Query time: {}'.format((t2 - t1) / len(queries)))
    print('Precision: {}'.format(float(score) / len(queries)))
    print('We considered a space of',len(dataset_copy),'3d points')
    print()
