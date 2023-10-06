#
# Basic introduction to SOM:
# https://medium.com/machine-learning-researcher/self-organizing-map-som-c296561e2117
#

#
# pip install python-Levenshtein
#

import pandas as pd
from Levenshtein import distance as levenshtein_distance
import subprocess
import random
import sys

print("=== Reading ARCEP data from data/2022_QoS_Metropole_data_habitations.csv === ")

# Read from the begining
#df = pd.read_csv('data/2022_QoS_Metropole_data_habitations.csv',sep=';',encoding='ISO-8859-1',low_memory=False,nrows=4)

# Read random rows
filename = "data/2022_QoS_Metropole_data_habitations.csv"
n = sum(1 for line in open(filename,encoding='ISO-8859-1')) - 1      #number of records in file (excludes header)
s = 70                                         #desired sample size
skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename,sep=';',encoding='ISO-8859-1',low_memory=False,skiprows=skip)

print('Column names of ARCEP data:')
for i in list(df.columns):
    print('\t',i)
#print(df.info())
#sys.exit()
    
print("=== Reading year 2021 ENEDIS data from of data/consommation-electrique-par-secteur-dactivite-commune.csv ===")
dff = (pd.read_csv('data/consommation-electrique-par-secteur-dactivite-commune.csv',sep=';',encoding='utf-8',low_memory=False)   [lambda x: (x['Année'] == 2021) & (x['CODE GRAND SECTEUR'] == 'RESIDENTIEL') ])#[lambda x: x['Code Commune'] == '41194'])
print('Column names of ENEDIS data (consomation):')
for i in list(dff.columns):
    print('\t',i)

print("=== Reading year 2021 ENEDIS data from of data/production-electrique-par-filiere-a-la-maille-commune.csv ===")
dfff = (pd.read_csv('data/production-electrique-par-filiere-a-la-maille-commune.csv',sep=';',encoding='utf-8',low_memory=False)   [lambda x: x['Année'] == 2021])#[lambda x: x['Code commune'] == '41194'])
print('Column names of ENEDIS data (production):')
for i in list(dfff.columns):
    print('\t',i)

print('=== Starting to generate new data ===')

# Compute Levenshtein
#mylist = []
#for row in df.itertuples(index=False):
#    if row[4].upper() not in mylist:
#        mylist.append(row[4].upper())
#print(mylist)
#str1=mylist[0]
#for i in mylist:
#    print('Levenshtein distance between',str1,'and',i,"=",levenshtein_distance(str1,i))


"""
-------- Message transféré --------
Sujet : 	RE: ENEDIS Open Data - Nouveau message de CHRISTOPHE CERIN
Date : 	Mon, 23 Jan 2023 14:52:33 +0000
De : 	OPENDATA <opendata@enedis.fr>
Pour : 	Christophe.cerin@univ-paris13.fr <Christophe.cerin@univ-paris13.fr>


Bonjour Monsieur Cerin,

Nous vous remercions pour l’intérêt que vous portez à notre open data.

Notre page « Bilan du Territoire » n’est pas requêtable en l’état, mais elle utilise plusieurs jeux de données qui le sont, en fonction de la maille géographique sélectionnée.

Pour la consommation, les jeux de données à requêter sont les suivants :

https://data.enedis.fr/explore/?sort=modified&q=Consommation+et+thermosensibilit%C3%A9&refine.keyword=Secteur+d%E2%80%99activit%C3%A9

Pour la production, les jeux de données à requêter sont les suivants :

https://data.enedis.fr/explore/?sort=modified&q=Production+%C3%A9lectrique+annuelle
 
Dans le cas où le projet que vous menez est un projet utilisant des données ouvertes et accessible à tous, vous pouvez le soumettre sur notre page « Réutilisations » :

https://data.enedis.fr/pages/reuse/
"""
 
#import geopy
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="OGD.AT-Lab")

import geopandas
import fiona

# lieu	situation	date	heure	operateur	Profil	rsrp	latitude	longitude	protocole	url	file_name	file_type	terminal	adresse	strate	sous_strate	page_charg�e_moins_5s	page_charg�e_moins_10s	d�bit_en_Mbit/s	video_en_qualit�_parfaite	video_en_qualit�_correcte	fichier_charg�_en_moins_de_30s	temps_en_secondes	delai_lancement_stream_s	lag_stream_s	accroche_5G	INSEE_DEP	INSEE_REG	NOM_DEP
# We choose : operateur, latitude, longitude, file_name (url), debit_en_Mbit/s, delai_lancement_stream_s 
# but we replae the operateur with a Levenshtein distance
for row in df.itertuples(index=False):
    if row[4] != '' and row[7] != '' and row[8] != '' and row[10] != '' and row[18] != '' and row[23] != '':

        myposition = str(row[7]) +','+str(row[8])

        # set your location

        location = geolocator.reverse(myposition)

        #print(location.address)
        #print((location.latitude, location.longitude))
        #print(location.raw)
        #print(location)
        mydict = dict(location.raw)
        #print('Commune :',mydict['address']['municipality'])
        #print('Code postal :',mydict['address']['postcode'])
        mycodepostal = str(mydict['address']['postcode'])[0:5]
        ville = str.upper(mydict['address']['municipality'])
        
        # Get the Code INSEE commune
        filenameINSEE = 'data/correspondance-code-insee-code-postal.geojson'
        myfile = open(filenameINSEE)
        #gdf = geopandas.read_file(file,where="postal_code='03200'",)
        mycond="postal_code=\'"+str(mycodepostal)+"\' and nom_comm=\'"+str(ville)+"\'"
        #print('Condition:',mycond)
        gdf = geopandas.read_file(myfile,where=mycond,)
        #print('GDF:',gdf)
        myfile.close()

        if gdf.empty:
            # Get the Code INSEE commune
            filenameINSEE = 'data/correspondance-code-insee-code-postal.geojson'
            myfile = open(filenameINSEE)
            #gdf = geopandas.read_file(file,where="postal_code='03200'",)
            mycond="postal_code=\'"+str(mycodepostal)+"\'"
            #print('Condition:',mycond)
            gdf = geopandas.read_file(myfile,where=mycond,)
            #print('GDF:',gdf)
            myfile.close()

        if gdf.empty:
            # Get the Code INSEE commune
            filenameINSEE = 'data/correspondance-code-insee-code-postal.geojson'
            myfile = open(filenameINSEE)
            #gdf = geopandas.read_file(file,where="postal_code='03200'",)
            mycond="nom_comm=\'"+str(ville)+"\'"
            #print('Condition:',mycond)
            gdf = geopandas.read_file(myfile,where=mycond,)
            #print('GDF:',gdf)
            myfile.close()

        if gdf.empty:
            continue

        #print('========')
        #print("Code INSEE commune :",gdf.at[0,'insee_com'])
        #print('========')

        #print(levenshtein_distance(str1,row[4]),';',row[7],';',row[8],';',gdf.at[0,'insee_com'],';',ville,';',row[10],';',row[18],';',row[23].replace(',','.'))

        # Explore the ENEDIS DATA (consommation)
        #print(dff.info())
        #'CODE GRAND SECTEUR' == RESIDENTIEL
        #'Code Commune' == gdf.at[0,'insee_com']
        #print(type(dff['Code Commune']))
        #print(type(gdf.at[0,'insee_com']))
        current_item_conso = dff[dff['Code Commune'] == int(gdf.at[0,'insee_com'])]
        if current_item_conso.empty:
            continue
        ConsoTotale  = current_item_conso.iloc[0]['Conso totale (MWh)']
        ConsoMoyenne = current_item_conso.iloc[0]['Conso moyenne (MWh)']
        #print('Conso Totale:',ConsoTotale,' <==> Conso Moyenne:',ConsoMoyenne)

        # Explore the ENEDIS DATA (production)
        #print(dffd.info())
        #'Code commune' == gdf.at[0,'insee_com']
        current_item_prod = dfff[dfff['Code commune'] == int(gdf.at[0,'insee_com'])]

        if current_item_prod.empty:
            continue
        
        ProdA  = current_item_prod.iloc[0]['Energie produite annuelle Photovoltaïque Enedis (MWh)']
        ProdB  = current_item_prod.iloc[0]['Energie produite annuelle Eolien Enedis (MWh)']
        ProdC  = current_item_prod.iloc[0]['Energie produite annuelle Hydraulique Enedis (MWh)']
        ProdD  = current_item_prod.iloc[0]['Energie produite annuelle Bio Energie Enedis (MWh)']
        ProdE  = current_item_prod.iloc[0]['Energie produite annuelle Cogénération Enedis (MWh)']
        ProdF  = current_item_prod.iloc[0]['Energie produite annuelle Autres filières Enedis (MWh)']

        # Analyse the URL
        result = subprocess.run(['python3', 'test_ecoindex.py', row[10]], stdout=subprocess.PIPE)
        #print(result.stdout.decode('utf-8'))
        
        print(row[4].upper(),';',row[7],';',row[8],';',gdf.at[0,'insee_com'],';',ville,';',result.stdout.decode('utf-8'),';',row[18],';',row[23].replace(',','.'),';',ConsoTotale,';',ConsoMoyenne,';',ProdA,';',ProdB,';',ProdC,';',ProdD,';',ProdE,';',ProdF,sep='')
        #pass
    else:
        print('ERROR: void column')
        sys.exit()
    #print(row)
