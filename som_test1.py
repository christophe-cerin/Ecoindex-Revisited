#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An example file to illustrate Self Organizing Maps
on som1.csv. This CSV file contains an
aggregation of data from ARCEP and ENEDIS and
computed data (eco_index on ARCEP URLs).

Usage:
  $ python3 som_test1.py
Output:
  som1_example.png: image representing the SOM

Author:
  christophe.cerin@univ-paris13.fr

Date:
  October 6, 2023
"""

from sklearn_som.som import SOM
from sklearn.datasets import make_blobs

# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
# Import libraries for plotting results
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
#import seaborn as sns

import hashlib

try:
    # Python 2; Python 3 will throw an exception here as bytes are required
    hashlib.md5('')
    def my_hash(s):
        return ord(hashlib.md5(s).digest()[0])
except TypeError:
    # Python 3; encode the string first, return sum of bytes
    def my_hash(s):
        res = 0
        for i in hashlib.sha512(s.encode('utf8')).digest():
            res += i
        return res
        #return str(hashlib.sha256(s.encode('utf8')).digest()[0]) + str(hashlib.sha256(s.encode('utf8')).digest()[1]) + str(hashlib.sha256(s.encode('utf8')).digest()[2])

#
# Load som_dataset and hcolumns of features and labels
#
som_dataset = pd.read_csv('som1.csv',sep=';',encoding='ISO-8859-1',low_memory=False)
# Extract the 17 colums we are interested in
som_dataset = som_dataset[['operateur','latitude','longitude','CP','ville','url','dom','req','size','eco_index','GreenHouseGaz','water','PageChargeeMoins5s','temps_en_secondes','Conso_totale_(MWh)','Conso_moyenne_(MWh)','Photovoltaique']].to_numpy()

#
# Replace operator, ville, and url attributes by an hash()
#
d = {}
for row in som_dataset:
    avant0 = row[0]
    avant4 = row[4]
    avant5 = row[5]
    row[0] = int(my_hash(row[0])) # operator
    row[4] = int(my_hash(row[4])) #ville
    row[5] = int(my_hash(row[5])) # url
    d[row[0]] = avant0
    d[row[4]] = avant4
    d[row[5]] = avant5

#
#
# Print the hashed values
#
for keys , values in d.items():
    print('key:',keys,'value:',values.encode("latin1").decode("utf-8", "strict"))
    #print('key: {0} value: {1}'.format(keys, values))

#
# Force all values to be float32
#
som_dataset = np.float32(som_dataset)
#print(som_dataset)
    
# Instantiate SOM from  python_som
# Selecting shape automatically (providing dataset for constructor)
# Using default decay and distance functions
# Using gaussian neighborhood function
# Using cyclic arrays in the vertical and horizontal directions

# We want 3 clusters (m=3)
my_som = SOM(m=5, n=1, dim=17, random_state=1234)
my_som.fit(som_dataset)

predictions = my_som.predict(som_dataset)

# Plot the results
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,7))
x = som_dataset[:,0]
y = som_dataset[:,1]
colors = ['red', 'green', 'blue', 'magenta','yellow','black']

ax[0].scatter(x, y)#, cmap=pltcolors.ListedColormap(colors))
ax[0].title.set_text('Actual Classes')
ax[1].scatter(x, y, c=predictions, cmap=pltcolors.ListedColormap(colors))
ax[1].title.set_text('SOM Predictions')
plt.savefig('som1_example.png')

#
# Utilization of susi: https://github.com/felixriese/susi/blob/main/examples/SOMClustering.ipynb
#
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix

plt.clf()

# Classify and plot
som = susi.SOMClustering(
    n_rows=30,
    n_columns=30
)
#som.fit(X)
som.fit(som_dataset)
print("SOM fitted!")

# Plot u-Matrix
u_matrix = som.get_u_matrix()
plot_umatrix(u_matrix, 30, 30)
#plt.show()
plt.savefig('umatrix.png')
plt.clf()

# Plot clusters
clusters = som.get_clusters(som_dataset)
plt.scatter(x=[c[1] for c in clusters], y=[c[0] for c in clusters], c=y, alpha=0.2)
plt.gca().invert_yaxis()
#plt.show()
plt.savefig('clusters.png')
plt.clf()

# Plot neighborhood distance matrix
plot_nbh_dist_weight_matrix(som)
#plt.show()
plt.savefig('nbh_dist_weight_matrix.png')
plt.clf()
