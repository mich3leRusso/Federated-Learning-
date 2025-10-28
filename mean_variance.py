import re
import numpy as np
import os

TAW = []
TAG = []
C = []
E = []
exp = 0

path = '/davinci-1/home/dmor/'
files = [f for f in os.listdir(path) if f.startswith('cifar100_CSIx3.o')]
print(files)
nome_file = path + files[0]

with open(nome_file, 'r', encoding='utf-8') as file:
    for riga in file:
        if riga[0] == 'T':
            if exp == 9:
                values = re.findall(r'\d+\.\d+|\d+', riga.strip())
                TAG.append(float(values[0]))
                E.append(float(values[1]))
                TAW.append(float(values[2]))
                C.append(float(values[3]))
                exp = -1
            exp = exp + 1
#TAG.append(0.348)
#TAW.append(0.706)

#TAG = TAG + [0.3911, 0.3993, 0.3898]
#TAW = TAW + [0.738, 0.7356, 0.729]
print(TAG)
print(TAW)
print(f'dati raccolti: {len(TAW)}')
print(f'TAW: {np.array(TAW).mean():.3f} ± {np.array(TAW).std():.3f}')
print(f'TAG: {np.array(TAG).mean():.3f} ± {np.array(TAG).std():.3f}')
print(f'E: {np.array(E).mean():.3f} ± {np.array(E).std():.3f}')
print(f'C: {np.array(C).mean():.3f} ± {np.array(C).std():.3f}')
