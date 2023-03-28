#!/usr/bin/env python3
"""
An example file to illustrate Self Organizing Maps
on som_dataset.csv. This CSV file contains an
aggregation of data from ARCEP and ENEDIS and
computed data (Ecoindex on ARCEP URLs).

Usage:
  $ python3 som_test1.py
Output:
  ecoindex_example.png: image representing the SOM

Author:
  christophe.cerin@univ-paris13.fr

Date:
  March 28, 2023
"""

from sklearn_som.som import SOM

# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd
# Import libraries for plotting results
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import seaborn as sns

# Load som_dataset and hcolumns of features and labels
# The 84 constant is the number of lines of our CSV file
som_dataset = pd.read_csv('som_dataset.csv',sep=',',encoding='ISO-8859-1',low_memory=False,nrows=84)
# Extract the 11 colums we are interested in
som_dataset = som_dataset[['dom','size','requests','EcoIndex','GreenHouseGaz','water','PageChargeeMoins10s','temps_en_secondes','Conso totale (MWh)','Conso moyenne (MWh)','Photovoltaique']].to_numpy()

# Instantiate SOM from  python_som
# Selecting shape automatically (providing dataset for constructor)
# Using default decay and distance functions
# Using gaussian neighborhood function
# Using cyclic arrays in the vertical and horizontal directions

#my_som = SOM(m=3, n=1, dim=11)
# We want 3 clusters (m=3)
my_som = SOM(m=3, n=1, dim=11, random_state=1234)
my_som.fit(som_dataset)

predictions = my_som.predict(som_dataset)

# Plot the results
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,7))
x = som_dataset[:,0]
y = som_dataset[:,1]
colors = ['red', 'green', 'blue', 'magenta']

ax[0].scatter(x, y)#, cmap=pltcolors.ListedColormap(colors))
ax[0].title.set_text('Actual Classes')
ax[1].scatter(x, y, c=predictions, cmap=pltcolors.ListedColormap(colors))
ax[1].title.set_text('SOM Predictions')
plt.savefig('ecoindex_example.png')
