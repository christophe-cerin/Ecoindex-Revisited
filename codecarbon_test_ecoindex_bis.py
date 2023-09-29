#!/usr/bin/env python3
"""
Example file to illustrate the EcoIndex Computation through
the historical method, as explained in
 https://github.com/cnumr/GreenIT-Analysis

Extra works:
   - take into consideration the sizes of downloaded fonts,
     .css and .js files

$ python3 test_ecoindex.py http://www.google.com
and the reply is
http://www.google.com ; 80 ; 12 ; 19254 ; 90.97 ; 1.18 ; 1.77
with the URL, the DOM, requests, size, EcoIndex, Water, Gas emission
"""

# Import NumPy and Pandas for storing data
import numpy as np
import pandas as pd

import sys

# codecarbon
from codecarbon import EmissionsTracker

__author__ = "Christophe Cerin"
__copyright__ = "Copyright 2023"
__credits__ = ["Christophe Cerin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christophe Cerin"
__email__ = "christophe.cerin@univ-paris13.fr"
__status__ = "Experimental"


quantiles_dom = [0, 47, 75, 159, 233, 298, 358, 417, 476, 537, 603, 674, 753, 843, 949, 1076, 1237, 1459, 1801, 2479, 594601]
quantiles_req = [0, 2, 15, 25, 34, 42, 49, 56, 63, 70, 78, 86, 95, 105, 117, 130, 147, 170, 205, 281, 3920]
quantiles_size = [0, 1.37, 144.7, 319.53, 479.46, 631.97, 783.38, 937.91, 1098.62, 1265.47, 1448.32, 1648.27, 1876.08, 2142.06, 2465.37, 2866.31, 3401.59, 4155.73, 5400.08, 8037.54, 223212.26]


def computeEcoIndex(dom,req,size):
    q_dom = computeQuantile(quantiles_dom,dom)
    q_req = computeQuantile(quantiles_req,req)
    q_size= computeQuantile(quantiles_size,size)
    return 100 - 5 * (3*q_dom + 2*q_req + q_size)/6

def computeQuantile(quantiles,value):
    for i in range(1,len(quantiles)):
        if value < quantiles[i]:
            return (i -1 + (value-quantiles[i-1])/(quantiles[i] -quantiles[i-1]))
    return len(quantiles) - 1

def getEcoIndexGrade(ecoIndex):
    if (ecoIndex > 80):
        return "A"
    if (ecoIndex > 70):
        return "B"
    if (ecoIndex > 55):
        return "C"
    if (ecoIndex > 40):
        return "D"
    if (ecoIndex > 25):
        return "E"
    if (ecoIndex > 10):
        return "F"
    return "G"


def computeGreenhouseGasesEmissionfromEcoIndex(ecoIndex):
    return '{:.2f}'.format(2 + 2 * (50 - ecoIndex) / 100)

def computeWaterConsumptionfromEcoIndex(ecoIndex):
    return '{:.2f}'.format(3 + 3 * (50 - ecoIndex) / 100)


#
# Main
#
if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Bad number of argument. Require no parameter!")
        print('usage: python3 codecarbon_test_ecoindex_bis.py')
        exit()

    # Number of rows to read in the dataset
    my_nrows = 100000
        
    dataset = pd.read_csv('url_4ecoindex_dataset.csv',sep=';',encoding='utf-8',usecols=['dom', 'request', 'size'],low_memory=False,nrows=my_nrows)
    # normalize the 3rd column => divide by 1024 to convert it in KB
    v = np.array([1,1,1024])
    dataset = dataset / v
    # Filter nul values
    dataset = dataset[(dataset['dom'] > 0) & (dataset['request'] > 0) & (dataset['size'] > 0) ]
    #
    # Convert the dataset to a numpy array
    #
    dataset = dataset.to_numpy()
    dataset = dataset.astype(np.float32)

    
    tracker = EmissionsTracker()
    tracker.start()

    for dom, req, size in dataset:
        #print(dom,req,size)
        EcoIndex = computeEcoIndex(dom,req,size)

    # stop codecarbon
    emissions: float = tracker.stop()
                
    print(f"Emissions: {emissions} kg for {my_nrows} URLs")
