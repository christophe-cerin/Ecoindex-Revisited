#!/usr/bin/env python3
"""
Example file to illustrate the eco_index Computation through
the Random Projection Method.

$ python3 random_projection.py
"""

import numpy as np
#np.random.seed(22409)
#print('Seed: ',np.random.get_state())

__author__ = "Christophe Cerin"
__copyright__ = "Copyright 2022"
__credits__ = ["Christophe Cerin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christophe Cerin"
__email__ = "christophe.cerin@univ-paris13.fr"
__status__ = "Experimental"

nbits = 16  # number of hyperplanes and binary vals to produce
d = 3       # vector dimensions


# create a set of nbits hyperplanes, with 3 dimensions

plane_norms = np.asarray([[ 0.17948202, -0.02331933,  0.29259013],
                          [ 0.47583352,  0.0112268 ,  0.19343756],
                          [-0.20643599,  0.03985945, -0.12196869],
                          [-0.45649158,  0.35920924, -0.27687209],
                          [ 0.12306751,  0.10212248,  0.43044841],
                          [-0.21039116, -0.13446219, -0.17234   ],
                          [-0.26083396, -0.4709955 ,  0.49507483],
                          [ 0.41719436,  0.13956593, -0.18032168],
                          [ 0.05571976, -0.42167403, -0.41292161],
                          [-0.49402616,  0.17677883, -0.00967637],
                          [-0.47385351,  0.27802129, -0.03447816],
                          [ 0.38146016, -0.13643649, -0.12757649],
                          [-0.13217846, -0.16424901,  0.08194174],
                          [ 0.46222133, -0.09348531,  0.48727933],
                          [ 0.08785052,  0.34677379, -0.41145532],
                          [ 0.14556619,  0.40167101,  0.25916261]])

plane_norms = np.asarray([[-0.33836887,  0.16609725,  0.43031704],
               [ 0.20818225, -0.39693546,  0.16480965],
               [ 0.1756493 , -0.39100124, -0.41224128],
               [ 0.33428631,  0.24626426, -0.42406262],
               [ 0.41044446,  0.37130889,  0.06660722],
               [-0.36221507,  0.20728521,  0.42821918],
               [ 0.42439602, -0.04654094,  0.27342524],
               [-0.32000925, -0.46403755,  0.25565296],
               [-0.31157464, -0.37811455,  0.13808374],
               [ 0.01019945, -0.36775938,  0.3057339 ],
               [ 0.08775265, -0.46987834, -0.17833812],
               [-0.45252127,  0.4717049 ,  0.43658814],
               [ 0.41793706, -0.18490246,  0.00810281],
               [-0.21151386,  0.40747742, -0.26328739],
               [-0.4624344,  -0.00386414, -0.48729842],
               [ 0.07965511, -0.11191503,  0.45014716]])

plane_norms = np.asarray([[-0.11493204,  0.26385645,  0.00357638],
                          [ 0.29657791, -0.09066585, -0.27606453],
                          [ 0.17699318,  0.2594418 ,  0.14981969],
                          [ 0.24734464,  0.4981493 ,  0.39143419],
                          [ 0.25919265,  0.35431568,  0.33106571],
                          [-0.35382281, -0.06827855,  0.36042934],
                          [-0.25952131,  0.01551095,  0.49060403],
                          [-0.10195086,  0.00468789, -0.32134607],
                          [ 0.38598717, -0.19451953, -0.00591825],
                          [ 0.41161074,  0.44527381,  0.23691653],
                          [ 0.21410202, -0.02805541, -0.42033325],
                          [ 0.06522854, -0.37576908, -0.33113413],
                          [ 0.27635687, -0.4573536 ,  0.45239895],
                          [ 0.01489645,  0.27367119, -0.07940135],
                          [-0.40042039, -0.36519427,  0.22766641],
                          [-0.0158775 ,  0.1348194 , -0.09412211]])

plane_norms = np.asarray([[-0.36612655,  0.23971237,  0.45920848],
                          [ 0.23462451, -0.27813106,  0.04475579],
                          [ 0.09850238, -0.47477873, -0.08123591],
                          [ 0.28368416,  0.06969451,  0.20213558],
                          [-0.29403178,  0.44435421, -0.42602345],
                          [ 0.45747935, -0.11495228,  0.40599255],
                          [-0.30813472,  0.0263537 ,  0.00091633],
                          [ 0.25333324, -0.17381389, -0.24191178],
                          [ 0.22277613,  0.49841663,  0.01225106],
                          [ 0.10380193,  0.00371674, -0.01286281],
                          [-0.11970782,  0.32391573,  0.22044143],
                          [-0.32315305,  0.37839939, -0.13254394],
                          [-0.35942265,  0.36982087,  0.35214944],
                          [-0.1549669 , -0.46222995, -0.17034234],
                          [-0.14603033, -0.25283692,  0.26728685],
                          [-0.32239571,  0.32577251, -0.26425989]])


plane_norms = np.asarray([[ 0.27673107, -0.47384733,  0.29105307],
                          [-0.49444111,  0.31371454,  0.28690729],
                          [ 0.45552663,  0.39542033,  0.46468427],
                          [-0.33812374,  0.44147975, -0.10924026],
                          [-0.23118018, -0.14931469, -0.32883606],
                          [-0.30982969, -0.33366807, -0.38858967],
                          [-0.15350551, -0.25089668,  0.34074077],
                          [-0.16569936, -0.13931132,  0.33815797],
                          [ 0.43945412, -0.05850384, -0.2411462 ],
                          [-0.34618155, -0.0013205 ,  0.27776988],
                          [-0.43539407, -0.25513099, -0.40797373],
                          [ 0.027216  , -0.04203971, -0.27095881],
                          [ 0.40969919,  0.0972698 ,  0.15798631],
                          [ 0.46808703, -0.26608765,  0.0995338 ],
                          [-0.04988305, -0.19283913,  0.02629862],
                          [-0.00097298,  0.37530183,  0.39326558]])

plane_norms = np.asarray([[ 0.2251249 ,  0.14437926,  0.3455753 ],
                          [-0.10159052,  0.33428272,  0.47959944],
                          [ 0.10342357,  0.01935221,  0.11617527],
                          [ 0.38999398,  0.11141542, -0.2586675 ],
                          [-0.38582652,  0.26240475,  0.35623112],
                          [ 0.06537124, -0.49027634, -0.42345763],
                          [-0.43143733,  0.03655048,  0.41205281],
                          [ 0.33625441, -0.23131361,  0.36325072],
                          [-0.39474928,  0.18521927, -0.36564689],
                          [-0.01422019,  0.19208727, -0.25837132],
                          [ 0.41302447, -0.42083791, -0.39670167],
                          [-0.40065976, -0.42283035,  0.46042274],
                          [-0.23034781, -0.11485182,  0.12385592],
                          [ 0.21681595,  0.38261832,  0.28784253],
                          [-0.24997801,  0.00581405, -0.13328182],
                          [-0.07256512, -0.2118799 ,  0.09576998]])

#plane_norms = np.random.rand(nbits, d) - .5
print('Plane-norms: ',plane_norms)
#print('Transpose: ',plane_norms.T)

a = np.asarray([1, 2, 3])
b = np.asarray([2, 1, 3])
c = np.asarray([0, 0, 0])
# calculate the dot product for each of these
a_dot = np.dot(a, plane_norms.T)
b_dot = np.dot(b, plane_norms.T)
c_dot = np.dot(c, plane_norms.T)
#print(a_dot)

# we know that a positive dot product == +ve side of hyperplane
# and negative dot product == -ve side of hyperplane
a_dot = a_dot > 0
b_dot = b_dot > 0
c_dot = c_dot > 0
#a_dot

# convert our boolean arrays to int arrays to make bucketing
# easier (although is okay to use boolean for Hamming distance)
a_dot = a_dot.astype(int)
b_dot = b_dot.astype(int)
c_dot = c_dot.astype(int)
#print(a_dot)
#print(b_dot)
#print(c_dot)

vectors = [a_dot, b_dot, c_dot]
buckets = {}
i = 0

import math
for i in range(len(vectors)):
    # convert from array to string
    hash_str = ''.join(vectors[i].astype(str))
    # create bucket if it doesn't exist
    if hash_str not in buckets.keys():
        buckets[hash_str] = []
    # add vector position to bucket
    buckets[hash_str].append(i)

#print(buckets)

#for key, value in buckets.items():
#    print('key: ',int(key,2),' ; --> ',value,' ; rank --> ',100.0 * int(key,2)/(math.pow(2,nbits)-1))

def compute_eco_index_random_projection(dom,req,size):
    a = np.asarray([dom, req, size])
    a_dot = np.dot(a, plane_norms.T)
    a_dot = a_dot > 0
    a_dot = a_dot.astype(int)
    #vectors = [a_dot]
    key = ''.join(a_dot.astype(str))
    #print(int(key,2))
    #return 100.0 - (100.0 * int(key,2)/(math.pow(2,nbits)-1))
    return (100.0 * int(key,2)/(math.pow(2,nbits)-1))

#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#Total sum of the bytes read:  19192
#Number of elements in DOM:  63
#Number of http requests:  12
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#URL:                                       http://www.google.com
#eco_index:                                  92.19
#eco_index Grade:                            A
#Greenhouse Gases Emission from eco_index:   1.16  (gCO2e)
#Water Consumption from eco_index:           1.73  (cl)
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#print('eco_index with LSH method: ',compute_eco_index_random_projection(63,12,19192/1024))
# eco_index with LSH method:  4.51262064848855
#print('eco_index with LSH method: ',compute_eco_index_random_projection(3*63,2*12,19192/1024))
# eco_index with LSH method:  4.51262064848855



#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#Total sum of the bytes read:  2215354
#Number of elements in DOM:  532
#Number of http requests:  57
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#URL:                                       https://www.pinecone.io/learn/locality-sensitive-hashing-random-projection/
#eco_index:                                  54.91
#eco_index Grade:                            D
#Greenhouse Gases Emission from eco_index:   1.90  (gCO2e)
#Water Consumption from eco_index:           2.85  (cl)
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#print('eco_index with LSH method: ',compute_eco_index_random_projection(532,57,2215354/1024))
# eco_index with LSH method:  95.48737935151145
#print('eco_index with LSH method: ',compute_eco_index_random_projection(3*532,2*57,2215354/1024))
# eco_index with LSH method:  88.85304147583736

#import numpy as np
#from sklearn import random_projection
#X = np.random.rand(100, 10000)
#transformer = random_projection.SparseRandomProjection()
#transformer = random_projection.GaussianRandomProjection()
#X_new = transformer.fit_transform(X)
#print(X_new.shape)
#(100, 3947)

#
# Calcul eco_index based on formula from web site www.eco_index.fr
#

quantiles_dom = [
    0, 47, 75, 159, 233, 298, 358,
    417, 476, 537, 603, 674, 753, 843,
    949, 1076, 1237, 1459, 1801, 2479, 594601
]
quantiles_req = [
    0, 2, 15, 25, 34, 42, 49,
    56, 63, 70, 78, 86, 95, 105,
    117, 130, 147, 170, 205, 281, 3920
]

quantiles_size = [
    0, 1.37, 144.7, 319.53, 479.46, 631.97, 783.38, 937.91,
    1098.62, 1265.47, 1448.32, 1648.27, 1876.08, 2142.06, 2465.37, 
    2866.31, 3401.59, 4155.73, 5400.08, 8037.54, 223212.26
]

def compute_eco_index(dom,req,size):
    q_dom = compute_quantile(quantiles_dom,dom)
    q_req = compute_quantile(quantiles_req,req)
    q_size= compute_quantile(quantiles_size,size)
    return 100 - 5 * (3*q_dom + 2*q_req + q_size)/6
           
def compute_quantile(quantiles,value):
    for i in range(1,len(quantiles)):
        if value < quantiles[i]:
            return (i -1 + (value-quantiles[i-1])/(quantiles[i] -quantiles[i-1]))
    return len(quantiles) - 1

sample_data = np.random.randint(size = (15,3), low = [20,10,1000], high = [50,30,20000])

#sample_data = np.random.randint(size = (15,3), low = [63,14, 22200], high = [64,15,22300])

for i in sample_data:
    eco = compute_eco_index(i[0],i[1],i[2]/1024)
    eco_lsh = compute_eco_index_random_projection(i[0],i[1],i[2])
    print(i,' eco_index: ',eco,' eco_index_Random_Projection: ',eco_lsh,' Diff: ',eco - eco_lsh)


def centroid_computation(arr):
    length = arr.shape[0]
    #print(length)
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x/length, sum_y/length, sum_z/length

#for i in sample_data:
#    print(i)
    
#print('Centroid: ',centroid_computation(sample_data))
#print('x = {.2%} y = {.2%} z = {.2%}'.format(float(sample_data[0]),float(sample_data[1]),float(sample_data[2])))

x,y,z = centroid_computation(sample_data)
print('x={:2.2f} y={:2.2f} z={:2.2f}'.format(float(x),float(y),float(z)))
