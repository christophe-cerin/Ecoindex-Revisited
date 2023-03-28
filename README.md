# EcoIndex-Revisited

## Introduction

In this project, we revisit the calculation method of the EcoIndex metric. This metric has been proposed to evaluate its absolute environmental performance from a given URL using a score out of 100 (higher is better). Our motivation comes from the fact that the calculation is based on both prior quantile calculations and weightings. We propose keeping only the weighting mechanism corresponding to documented and regularly available figures of the proportional breakdown of ICT's carbon footprint. 

This way, we will be able to follow, from year to year, the evolution of web requests from a carbon footprint point of view. For a URL, our new calculation method takes as parameters three weights and the three typical values of the EcoIndex (DOM size, number of HTTP/HTTPS requests, KB transferred) and returns an environmental performance score. 

We develop several ways to compute the score based on our new hypothesis, either using learning techniques (Locality Sensitive Hashing, K Nearest Neighbor) or matrix computation constitutes the project's first contribution. The second contribution corresponds to an experimental study that allows us to estimate the differences in results between the methods. The whole work allows us to observe the environmental performance of the WEB in a more generic way than with the initial method. 

Indeed, the initial process requires recalculating each quantile according to the value of the chosen weights. It is, therefore, necessary to launch a benchmark, the HTTP archive, for example, at each new weighting. Our approaches do not require a systematic switch to a benchmark; thus, it is more generic than the previously known one.

## Python codes and dataset explained

 - requirements.txt file serve as a list of items to be installed by pip, when using pip install. Files that use this format are often called “pip requirements.txt files”, since requirements.txt is usually what these files are named (although, that is not a requirement). So, to install the dependencies, run first `pip install -r requirements.txt`;

 - url_4ecoindex_dataset.csv is a dataset corresponding to more than 100k requests from the HTTParchive (a subset dated April 2022). This CSV file gives the URL, the DOM, request, and the size collected through the execution of test_ecoindex.py on the URL. On the same line, you get the EcoIndex, then the water consumption and the gas emission values;
- test_ecoindex.py implements the original EcoIndex; You get a CSV-like file with the URL, DOM, request, size, EcoIndex, water consumption, and gas emission;
```
$ python3 test_ecoindex.py http://www.google.fr
http://www.google.fr ; 80 ; 12 ; 19160 ; 90.97 ; 1.18 ; 1.77
```
- random_projection.py implements a random projection method for the EcoIndex. The EcoIndex is given by the rank of the bin receiving the projection. The code generates random samples, and we compute the historical EcoIndex, the new EcoIndex, and then the difference between the two;
```
$ python3 random_projection.py
Plane-norms:  [[ 0.2251249   0.14437926  0.3455753 ]
 [-0.10159052  0.33428272  0.47959944]
 [ 0.10342357  0.01935221  0.11617527]
 [ 0.38999398  0.11141542 -0.2586675 ]
 [-0.38582652  0.26240475  0.35623112]
 [ 0.06537124 -0.49027634 -0.42345763]
 [-0.43143733  0.03655048  0.41205281]
 [ 0.33625441 -0.23131361  0.36325072]
 [-0.39474928  0.18521927 -0.36564689]
 [-0.01422019  0.19208727 -0.25837132]
 [ 0.41302447 -0.42083791 -0.39670167]
 [-0.40065976 -0.42283035  0.46042274]
 [-0.23034781 -0.11485182  0.12385592]
 [ 0.21681595  0.38261832  0.28784253]
 [-0.24997801  0.00581405 -0.13328182]
 [-0.07256512 -0.2118799   0.09576998]]
[   41    10 18482]  EcoIndex:  94.19653572440276  EcoIndex_Random_Projection:  91.8425268940261  Diff:  2.354008830376671
[   46    10 15112]  EcoIndex:  93.94971253435023  EcoIndex_Random_Projection:  91.8425268940261  Diff:  2.1071856403241327
[   24    28 14929]  EcoIndex:  92.25771647830204  EcoIndex_Random_Projection:  91.8425268940261  Diff:  0.4151895842759501
[   23    29 14974]  EcoIndex:  92.12546728053375  EcoIndex_Random_Projection:  91.8425268940261  Diff:  0.2829403865076614
[   44    24 12971]  EcoIndex:  91.92722608680008  EcoIndex_Random_Projection:  91.8425268940261  Diff:  0.08469919277398219
[   45    12 11180]  EcoIndex:  93.7688189594573  EcoIndex_Random_Projection:  91.8425268940261  Diff:  1.9262920654312126
[  49   23 3386]  EcoIndex:  91.8101687710553  EcoIndex_Random_Projection:  91.8425268940261  Diff:  -0.032358122970791214
[  32   15 3451]  EcoIndex:  94.11957681502088  EcoIndex_Random_Projection:  91.8425268940261  Diff:  2.2770499209947843
[   49    14 19967]  EcoIndex:  93.1775632826618  EcoIndex_Random_Projection:  91.8425268940261  Diff:  1.3350363886357002
[  47   18 6663]  EcoIndex:  92.80346731355671  EcoIndex_Random_Projection:  91.8425268940261  Diff:  0.9609404195306155
[  37   18 4369]  EcoIndex:  93.34840712853818  EcoIndex_Random_Projection:  91.8425268940261  Diff:  1.505880234512091
[  41   19 4908]  EcoIndex:  92.96591415890795  EcoIndex_Random_Projection:  91.8425268940261  Diff:  1.1233872648818561
[   20    21 16709]  EcoIndex:  93.68259813659849  EcoIndex_Random_Projection:  91.8425268940261  Diff:  1.8400712425723924
[  35   10 8773]  EcoIndex:  94.57081062462156  EcoIndex_Random_Projection:  91.8425268940261  Diff:  2.7282837305954644
[  31   25 7241]  EcoIndex:  92.48458269614169  EcoIndex_Random_Projection:  91.8425268940261  Diff:  0.6420558021155927
x=37.60 y=18.40 z=10874.33
```
- lsh.py implements a Locality Sensitive Hashing (LSH) method for the EcoIndex. We use the Falconn package and select two random queries taken from the input. We search for these two inputs and compute the EcoIndex according to the LSH method. We first go through the k=3 nearest neighbors, compute the barycenter, and then the EcoIndex; 
```
$ python3 lsh.py
Normalizing the dataset
Done
Generating queries
Queries:  [array([ 97.,  21., 172.], dtype=float32), array([122.,  59.,  25.], dtype=float32)]
Done
Solving queries using linear scan
Done
Linear scan time: 0.06975744999999733 per query
Constructing the LSH table
Done
Construction time: 16.425820600001316
Choosing number of probes
21 -> 1.0
Done
21 probes
found:  [0.48846823 0.10575085 0.86614984]  -->  [ 88.  19. 156.]
Centroid of the k nearest neighbors: [93.84, 20.31111111111111, 166.40444444444444]
EcoIndex: 58.44
found:  [0.885314   0.42814365 0.18141681]  -->  [122.  59.  25.]
Centroid of the k nearest neighbors: [168.88444444444445, 81.68444444444444, 34.60888888888889]
EcoIndex: 57.75
Query time: 2.9719452999997884
Precision: 1.0
We considered a space of 11390625 3d points
```
-  collinearity.py implements a method considering the most collinear vector points with the query for the EcoIndex metric. First, we isolate candidate points and compute the centroid of these points. The EcoIndex is calculated as a 'relative position' for the centroid in the considered virtual space. The following example shows the query with Dom=1<span>&#215;</span>9, request=1<span>&#215;</span>8, and size=1<span>&#215;</span>15. Parameter 8 corresponds to the virtual space size, i.e., 8<sup>3</sup>=512, meaning that we deal with 512 points conceptually.
```
$ python3 collinearity.py 1 1 1 9 5 15 8
Arguments count: 8
Argument      0: collinearity.py
Argument      1: 1
Argument      2: 1
Argument      3: 1
Argument      4: 9
Argument      5: 5
Argument      6: 15
Argument      7: 8
Query          : [9, 5, 15]
Normalizing the dataset of length: 512
Dataset normalized
Final centroid: [1.140625, 0.671875, 1.890625]
EcoIndex: 98.07
Query time: 0.01154590000078315
We used a 3-d virtual space of 512 random 3d points
```

Anyway, please, read first the headers of Python programs for usage. You may also play with some internal variables.

## Analysis of the dataset

File `analysis_mj.ipynb` corresponds to a Jupyter notebook analyzing data over file `url_4ecoindex_dataset.csv`. It aims to check how different the new EcoIndex and the historical EcoIndex, faced quantiles updates. Visualization helps to quantify the differences throughout multiple techniques and metrics. File `analysis_mj.pdf` is the generated PDF file obtained after running the analysis.

## Self-Organizing Map (SOM)

File `som_test1.py` generates a PNG image corresponding to a self-organizing map. SOM is used in the exploration phase, and it clusters data. The dataset used in this example is `som_dataset.csv`, built from ARCEP 2022_QoS_Metropole_data_habitations.csv` and ENEDIS (`consommation-electrique-par-secteur-dactivite-commune.csv` ; `production-electrique-par-filiere-a-la-maille-commune.csv`) datasets. Some data from these datasets are combined with EcoIndex data (DOM, request, size...) for the URL. This example aims to demonstrate that we can deal with more than 10 attributes related to energy. Check with the header of `som_dataset.csv` to appreciate the metrics we deal with, and also with ARCEP and ENEDIS for their open data (`https://data.enedis.fr/explore/dataset/consommation-electrique-par-secteur-dactivite-commune/export/` ; `https://data.enedis.fr/explore/dataset/production-electrique-par-filiere-a-la-maille-commune/export/?sort=annee` and `https://files.data.gouv.fr/arcep_donnees/mobile/mesures_qualite_arcep/2022/Metropole/`). 





