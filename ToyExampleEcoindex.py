#
# Toy example fo EcoIndex computation: sampling 5 lines of
# the url_4ecoindex_dataset.csv dataset, compute the Ecoindx
# scores for the urls, and draw a figure.
#

import subprocess
import sys
import random

Nb      = 5

x = []; y = []

with open(r"url_4ecoindex_dataset.csv", 'r') as fp:
    NbLines = len(fp.readlines())
    print('Total lines:', NbLines) # 8

l = random.sample(range(0, NbLines), Nb)

with open(r"url_4ecoindex_dataset.csv", 'r') as fp:
    for i, line in enumerate(fp):
        if i in l:
            # process line
            s = line.split(';')[0]
            #s = 'http://www.google.com'
            # create two files to hold the output and errors, respectively
            with open('out.txt','w+') as fout:
                with open('err.txt','w+') as ferr:
                    out=subprocess.call(["python3",'test_ecoindex.py',s],stdout=fout,stderr=ferr)
                    # reset file to read from it
                    fout.seek(0)
                    # save output (if any) in variable
                    output=fout.read()

                    # reset file to read from it
                    ferr.seek(0) 
                    # save errors (if any) in variable
                    errors = ferr.read()

            if output:
                url = output.split(';')[0].strip()
                score = float('%.2f' % float(output.split(';')[4].strip()))
                print("Resultat execution commande : ",url,score)
                #print("Errors :",errors)
                #x.append('\n'.join(url))
                x.append(url)
                y.append(score)

print('x coordinates:',x)
print('y coordinates:',y)

import matplotlib.pyplot as plt

#plt.figure(figsize=(9, 3))

plt.figure()
plt.xticks(rotation = 45)

#plt.subplot(131)
plt.bar(x, y)
#plt.subplot(132)
#plt.scatter(x, y)
#plt.subplot(133)
#plt.plot(x, y)
plt.suptitle('EcoIndex Toy Example')
ax = plt.gca()
plt.draw()

for tick in ax.get_xticklabels():
    tick.set_rotation(-45)

plt.tight_layout()
plt.show()
