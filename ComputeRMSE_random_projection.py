import math
# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
# Import libraries for plotting results
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load dataset
if not myCSV:
    print('========= READING DATASET ============')

som_dataset = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['dom', 'request', 'size','EcoIndex'],low_memory=False,nrows=my_nrows)
# normalize the 3rd column => divide by 1024 to convert it in KB
v = np.array([1,1,1024,1])
som_dataset = som_dataset / v
# Filter nul values
som_dataset = som_dataset[(som_dataset['dom'] > 0) & (som_dataset['request'] > 0) & (som_dataset['size'] > 0) ]
# Keep historical EcoIndex values
historical = som_dataset['EcoIndex']

# normalize the 1st and 2nd columns => weights = 3 and 2 => mimic the historical EcoIndex
#v = np.array([3,2,1])
#som_dataset = som_dataset * v
# centering the dataset
#center = np.mean(som_dataset, axis=0)
#som_dataset -= center

#historical = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['EcoIndex'],low_memory=False,nrows=my_nrows)

# Normalize all the lenghts, since we care about the cosine similarity.
#print(som_dataset.dtypes)
#assert som_dataset.dtypes == np.float32
#print('Normalizing the dataset')
#som_dataset /= np.linalg.norm(som_dataset, axis=1).reshape(-1, 1)
#print('dataset normalized: ',dataset)
#sns.load_dataset('som_dataset')
if not myCSV:
    print('========= END READING ================')

import sys
sys.path.append('LSHash/lshash')
from lshash import *
from storage import *

average_RMSE = []
min_RMSE = 100000000
max_RMSE = -100000000

for x,y in zip(som_dataset.values,historical.to_numpy()):
    for foo in range(1,25):
        #
        # We build a 3-D space and we keep 7 bits for the bitstring, hence 128 keys/buckets
        #
        nb_bits = 19
        lsh = LSHash(nb_bits, 3)

        y_actual = []
        y_predicted = []
 
        query = [x[0],x[1],x[2]]
        for i in lsh.index(query):
            #query += center
            #myList = query.tolist()
            #myList = [int(x) for x in myList]
            predicted = (int(i,2)/math.pow(2,nb_bits))*100.0
            #print('Bucket id:',int(i,2),'for input:',myList,'EcoIndex LSH:',predicted,'Hitorical:',y[0])
            y_actual.append(y)
            y_predicted.append((int(i,2)/math.pow(2,nb_bits))*100.0)

    if not myCSV:
        print('EcoIndex: {:.2f}'.format(sum(y_predicted)/len(y_predicted)),'; Historical EcoIndex: {:.2f}'.format(y))
    else:
        print(x[0],';',x[1],';',x[2],'; {:.2f}'.format(y),'; {:.2f}'.format(sum(y_predicted)/len(y_predicted)),)

    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
    RMSE = math.sqrt(MSE)
    average_RMSE.append(RMSE)
    min_RMSE = min(min_RMSE,RMSE)
    max_RMSE = max(max_RMSE,RMSE)

if not myCSV:
    from statistics import mean
    print("Average Root Mean Square Error:",mean(average_RMSE))
    print("Min Root Mean Square Error:",min_RMSE)
    print("Max Root Mean Square Error:",max_RMSE)

