import numpy as np

items = np.arange(12)
rel = np.zeros(len(items))
for i in range(len(rel)):
    rel[i] = 1-(0.01)*i

g1 = [0,6,1,7,2,8,3,9,4,10,5,11]
g2 = [0,1,4,5,2,6,7,8,3,9,10,11]

idcg = np.zeros(len(rel)+1)
idcg[0] = 0
for i in range(1,len(rel)+1,1):
    idcg[i] = idcg[i-1] + (2**rel[i-1])/np.log2(i+1)
print(idcg)
dcg1 = np.zeros(len(rel)+1)
dcg1[0] = 0
for i in range(1,len(rel)+1,1):
    dcg1[i] = dcg1[i-1] + (2**rel[g1[i-1]])/np.log2(i+1)
print(dcg1)
dcg2 = np.zeros(len(rel)+1)
dcg2[0] = 0
for i in range(1,len(rel)+1,1):
    dcg2[i] = dcg2[i-1] + (2**rel[g2[i-1]])/np.log2(i+1)
print(dcg2)
for i in range(1, len(rel), 1):
    print(i, dcg1[i]/idcg[i], dcg2[i]/idcg[i])