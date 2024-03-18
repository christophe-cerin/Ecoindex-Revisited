import math
# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
# Import libraries for plotting results
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

# Boolean indicating if we generate a CSV format or not.
# In this last case we print the RMSE between the historical
# eco_index and the one computed with the 'colinearity method'.
myCSV = False

# Nomber of lines to read in the input csv files
my_nrows = 102000

# Load dataset
if not myCSV:
    print('========= READING DATASET ============')

som_dataset = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['dom', 'request', 'size','eco_index'],low_memory=False,nrows=my_nrows)
# normalize the 3rd column => divide by 1024 to convert it in KB
v = np.array([1,1,1024,1])
som_dataset = som_dataset / v
# Filter nul values
som_dataset = som_dataset[(som_dataset['dom'] > 0) & (som_dataset['request'] > 0) & (som_dataset['size'] > 0) ]
#
# We find the outliers and we cancel them from the inputs
#
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.5,random_state=0).fit_predict(som_dataset[['dom','request','size']])
#print(np.where(clf == -1))
#print(type(np.where(clf == -1)[0].tolist()))
func = som_dataset.index
tt = list(map(lambda x: som_dataset.index[x],np.where(clf == -1)[0].tolist()))

#for i in np.where(clf == -1)[0].tolist():
#    print(som_dataset.iloc[[i]])

som_dataset.drop(tt, axis=0,inplace = True)
#
# update my_nrows
my_nrows, ncols = som_dataset.shape

#
# We keep a copy
#
historical = som_dataset['eco_index']

if not myCSV:
    print('========= END READING ================')

t1 = timeit.default_timer()

# Prime numbers
#p1,p2,p3 = 73856093, 19349663, 83492791

p = [0, 0, 0]
my_max = som_dataset.max(axis=0)

def my_distance_euclidienne_max(b,l):
    return sum([(a-b)**2 for a,b in zip(b,l)])

lower, upper = 0, my_distance_euclidienne_max(p,[my_max['dom'],my_max['request'],my_max['size']])

#print(lower,upper,[my_max['dom'],my_max['request'],my_max['size']])

def my_distance_euclidienne(b,l):
    d = sum([(a-b)**2 for a,b in zip(b,l)])
    return [100*(x / (lower + (upper - lower))) for x in [d]][0]

xdata= []
ydata = []
zdata = []

import sys

average_RMSE = []
min_RMSE = 1000000000
max_RMSE = -1000000000

if myCSV:
    print("dom ; request ; size ; historical ; distance")

for x,y in zip(som_dataset.values,historical.to_numpy()):
    for foo in range(1,2):
        
        y_actual = []
        y_predicted = []
 
        query = [x[0],x[1],x[2]]
        i = 100.0 - my_distance_euclidienne(p,query)

        y_actual.append(y)
        y_predicted.append(i)

    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
    RMSE = math.sqrt(MSE)
    average_RMSE.append(RMSE)
    min_RMSE = min(min_RMSE,RMSE)
    max_RMSE = max(max_RMSE,RMSE)
    
    xdata.append(x[0])
    ydata.append(x[1])
    zdata.append(x[2])
    
    XX = sum(y_predicted)/len(y_predicted)
    if myCSV:
        print(int(x[0]),';',int(x[1]),';','{:.2f}'.format(x[2]),';', '{:.2f}'.format(y),';', '{:.2f}'.format(XX))

t2 = timeit.default_timer()

if not myCSV:
    from statistics import mean
    print("Average Root Mean Square Error:",mean(average_RMSE))
    print("Min Root Mean Square Error:",min_RMSE)
    print("Max Root Mean Square Error:",max_RMSE)
    print('Compute time: {} per entry of the dataset'.format((t2 - t1) / my_nrows))
    print('Total completion time: {}'.format((t2 - t1)))
    
#
# if you do (not) want to plot your dataset, please (un)comment le following lines
#
#from mpl_toolkits import mplot3d
#import matplotlib.pyplot as plt

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

# Data for three-dimensional scattered points
#ax.scatter3D(xdata, ydata, zdata, c='red', marker='o');

#for angle in range(0, 360):
#   ax.view_init(30, angle)
#   plt.draw()
#   plt.pause(.001)

#plt.show()



