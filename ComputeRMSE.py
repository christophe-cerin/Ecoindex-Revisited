import math
# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
# Import libraries for plotting results
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset and columns of features and labels
print('========= READING DATASET ================')

my_nrows = 108000

som_dataset = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['dom', 'request', 'size'],low_memory=False,nrows=my_nrows)
# normalize the 3rd column => divide by 1024 to convert it in KB
v = np.array([1,1,1024])
som_dataset = som_dataset / v
# centering the dataset
center = np.mean(som_dataset, axis=0)
som_dataset -= center

historical = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['EcoIndex'],low_memory=False,nrows=my_nrows)

# Normalize all the lenghts, since we care about the cosine similarity.
#print(som_dataset.dtypes)
#assert som_dataset.dtypes == np.float32
#print('Normalizing the dataset')
#som_dataset /= np.linalg.norm(som_dataset, axis=1).reshape(-1, 1)
#print('dataset normalized: ',dataset)
#sns.load_dataset('som_dataset')
print('========= END READING ================')

import sys
sys.path.append('LSHash/lshash')
from lshash import *
from storage import *

average_RMSE = []
min_RMSE = 10000
max_RMSE = -10000

for foo in range(1,25):
    #
    # We build a 3-D space and we keep 7 bits for the bitstring, hence 128 keys/buckets
    #
    nb_bits = 7
    lsh = LSHash(nb_bits, 3)

    """
    for i in lsh.index([1,2,3,4,5,6,7,8]):
        print('Bucket id:',int(i,2))
    for i in lsh.index([2,3,4,5,6,7,8,9]):
        print('Bucket id:',int(i,2))
    for i in lsh.index([10,12,99,1,5,31,2,3]):
        print('Bucket id:',int(i,2))
    print(lsh.query([1,2,3,4,5,6,7,7]))
    """

    #print(type(som_dataset.values))
    #print(type(som_dataset.values))
    #print(type(historical.to_numpy()))

    y_actual = []
    y_predicted = []
 
    for x,y in zip(som_dataset.values,historical.to_numpy()):
        query = [x[0],x[1],x[2]]
        for i in lsh.index(query):
            query += center
            myList = query.tolist()
            myList = [int(x) for x in myList]
            predicted = (int(i,2)/math.pow(2,nb_bits))*100.0
            #print('Bucket id:',int(i,2),'for input:',myList,'EcoIndex LSH:',predicted,'Hitorical:',y[0])
            y_actual.append(y[0])
            y_predicted.append((int(i,2)/math.pow(2,nb_bits))*100.0)

    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
    RMSE = math.sqrt(MSE)
    average_RMSE.append(RMSE)
    min_RMSE = min(min_RMSE,RMSE)
    max_RMSE = max(max_RMSE,RMSE)

from statistics import mean
print("Average Root Mean Square Error:",mean(average_RMSE))
print("Min Root Mean Square Error:",min_RMSE)
print("Max Root Mean Square Error:",max_RMSE)

