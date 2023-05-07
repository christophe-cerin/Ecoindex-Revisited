import math
# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
# Import libraries for plotting results
import matplotlib.pyplot as plt
import seaborn as sns

print('========= RANDOM PROJECTION VERSUS COLLINEARITY ============')

random_projection = pd.read_csv('random_projection.csv',sep=';',encoding='ASCII',low_memory=False)

collinearity = pd.read_csv('collinearity.csv',sep=';',encoding='ASCII',low_memory=False)

average_RMSE = []
min_RMSE = 100000000
max_RMSE = -100000000

for x,y in zip(random_projection.to_numpy(),collinearity.to_numpy()):

    if x[0] == y[0] and x[1] == y[1] and math.trunc(x[2]) == math.trunc(y[2]) and x[3] == y[3]:
        actual = x[4]
        predicted = y[4]
 
        MSE = np.square(np.subtract(actual,predicted))
 
        #print(x[4],y[4],MSE)

        RMSE = math.sqrt(MSE)
        average_RMSE.append(RMSE)
        min_RMSE = min(min_RMSE,RMSE)
        max_RMSE = max(max_RMSE,RMSE)
    else:
        print("Error on",x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3])
        break

from statistics import mean
print("Average Root Mean Square Error:",mean(average_RMSE))
print("Min Root Mean Square Error:",min_RMSE)
print("Max Root Mean Square Error:",max_RMSE)

print('========= RANDOM PROJECTION VERSUS LSH KNN ============')

lsh_knn = pd.read_csv('lsh_knn.csv',sep=';',encoding='ASCII',low_memory=False)

random_projection = pd.read_csv('random_projection.csv',sep=';',encoding='ASCII',low_memory=False,nrows=len(lsh_knn.index))

average_RMSE = []
min_RMSE = 100000000
max_RMSE = -100000000

for x,y in zip(random_projection.to_numpy(),lsh_knn.to_numpy()):

    if x[0] == y[0] and x[1] == y[1] and math.trunc(x[2]) == math.trunc(y[2]) and x[3] == y[3]:
        actual = x[4]
        predicted = y[4]
 
        MSE = np.square(np.subtract(actual,predicted))
 
        #print(x[4],y[4],MSE)

        RMSE = math.sqrt(MSE)
        average_RMSE.append(RMSE)
        min_RMSE = min(min_RMSE,RMSE)
        max_RMSE = max(max_RMSE,RMSE)
    else:
        print("Error on",x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3])
        break

from statistics import mean
print("Average Root Mean Square Error:",mean(average_RMSE))
print("Min Root Mean Square Error:",min_RMSE)
print("Max Root Mean Square Error:",max_RMSE)

print('========= COLLINEARITY VERSUS LSH KNN ============')

lsh_knn = pd.read_csv('lsh_knn.csv',sep=';',encoding='ASCII',low_memory=False)

collinearity = pd.read_csv('collinearity.csv',sep=';',encoding='ASCII',low_memory=False,nrows=len(lsh_knn.index))

average_RMSE = []
min_RMSE = 100000000
max_RMSE = -100000000

for x,y in zip(collinearity.to_numpy(),lsh_knn.to_numpy()):

    if x[0] == y[0] and x[1] == y[1] and math.trunc(x[2]) == math.trunc(y[2]) and x[3] == y[3]:
        actual = x[4]
        predicted = y[4]
 
        MSE = np.square(np.subtract(actual,predicted))
 
        #print(x[4],y[4],MSE)

        RMSE = math.sqrt(MSE)
        average_RMSE.append(RMSE)
        min_RMSE = min(min_RMSE,RMSE)
        max_RMSE = max(max_RMSE,RMSE)
    else:
        print("Error on",x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3])
        break

from statistics import mean
print("Average Root Mean Square Error:",mean(average_RMSE))
print("Min Root Mean Square Error:",min_RMSE)
print("Max Root Mean Square Error:",max_RMSE)

print('========= DISTANCE VERSUS HISTORICAL ============')

my_distance = pd.read_csv('euclidean_distance.csv',sep=';',encoding='ASCII',low_memory=False)

average_RMSE = []
min_RMSE = 100000000
max_RMSE = -100000000

for x in zip(my_distance.to_numpy()):
    
    actual = x[0][3]
    predicted = x[0][4]
 
    MSE = np.square(np.subtract(actual,predicted))
 
    RMSE = math.sqrt(MSE)
    average_RMSE.append(RMSE)
    min_RMSE = min(min_RMSE,RMSE)
    max_RMSE = max(max_RMSE,RMSE)

from statistics import mean
print("Average Root Mean Square Error:",mean(average_RMSE))
print("Min Root Mean Square Error:",min_RMSE)
print("Max Root Mean Square Error:",max_RMSE)

