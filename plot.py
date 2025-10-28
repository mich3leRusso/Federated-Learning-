import re
import numpy as np
import os
import matplotlib.pyplot as plt
import ast

filename = ('cifar100_all_2')

medie_TAG = []
medie_TAW = []
medie_T = []

std_TAG = []
std_TAW = []
std_T = []

path = '/davinci-1/home/dmor/'
files = [f for f in os.listdir(path) if f.startswith(filename+'.o')]
print(files)
nome_file = path + files[0]
p=0
with open(nome_file, 'r', encoding='utf-8') as file:
    for riga in file:
        if riga[0] == 'n':
            numeri = re.findall(r'-?\d+(?:[\.,]\d+)?', riga)
            medie_TAG.append(float(numeri[1]))
            std_TAG.append(float(numeri[2]))
            medie_TAW.append(float(numeri[3]))
            std_TAW.append(float(numeri[4]))
            medie_T.append(float(numeri[5]))
            std_T.append(float(numeri[6]))
        if riga[0] == 'S':
            medie_TAG = []
            medie_TAW = []
            medie_T = []
            std_TAG = []
            std_TAW = []
            std_T = []
        if (riga[0] == 'N') & (len(medie_T)>0):
            l1 = len(medie_T)-1
            l2 = int(l1/2)
            print('\naug = 0')
            print(f'TAW: {medie_TAW[0]: .2f} ± {std_TAW[0]: .2f}')
            #print(f'T: {medie_T[0]: .1f} ± {std_T[0]: .1f}')
            print(f'TAG: {medie_TAG[0]: .2f} ± {std_TAG[0]: .2f}')
            print(f'\naug = {l2}')
            print(f'TAW: {medie_TAW[l2]: .2f} ± {std_TAW[l2]: .2f}')
            #print(f'T: {medie_T[l2]: .1f} ± {std_T[l2]: .1f}')
            print(f'TAG: {medie_TAG[l2]: .2f} ± {std_TAG[l2]: .2f}')
            print(f'\naug = {l1}')
            print(f'TAW: {medie_TAW[l1]: .2f} ± {std_TAW[l1]: .2f}')
            #print(f'T: {medie_T[l1]: .1f} ± {std_T[l1]: .1f}')
            print(f'TAG: {medie_TAG[l1]: .2f} ± {std_TAG[l1]: .2f}')
            print('\n')

l1 = len(medie_T)-1
l2 = int(l1/2)
print('\naug = 0')
print(f'TAW: {medie_TAW[0]: .2f} ± {std_TAW[0]: .2f}')
#print(f'T: {medie_T[0]: .1f} ± {std_T[0]: .1f}')
print(f'TAG: {medie_TAG[0]: .2f} ± {std_TAG[0]: .2f}')
print(f'\naug = {l2}')
print(f'TAW: {medie_TAW[l2]: .2f} ± {std_TAW[l2]: .2f}')
#print(f'T: {medie_T[l2]: .2f} ± {std_T[l2]: .2f}')
print(f'TAG: {medie_TAG[l2]: .2f} ± {std_TAG[l2]: .2f}')
print(f'\naug = {l1}')
print(f'TAW: {medie_TAW[l1]: .2f} ± {std_TAW[l1]: .2f}')
#print(f'T: {medie_T[l1]: .1f} ± {std_T[l1]: .1f}')
print(f'TAG: {medie_TAG[l1]: .2f} ± {std_TAG[l1]: .2f}')

'''medie_r = []
medie_TAW_r = []
std_r = []
std_TAW_r = []
files = [f for f in os.listdir(path) if f.startswith(filename+'_rot.o')]
nome_file = path + files[0]
p=0
with open(nome_file, 'r', encoding='utf-8') as file:
    for riga in file:
        if riga[0] == 'n':
            numeri = re.findall(r'-?\d+(?:[\.,]\d+)?', riga)
            medie_r.append(float(numeri[1]))
            std_r.append(float(numeri[2]))
            medie_TAW_r.append(float(numeri[3]))
            std_TAW_r.append(float(numeri[4]))
        if riga[0] == 'S':
            medie_r = []
            medie_TAW_r = []
            std_r = []
            std_TAW_r = []

medie_r = np.array(medie_r)
std_r = np.array(std_r)
medie = np.array(medie)
std = np.array(std)'''

'''plt.plot(range(1,len(medie)), medie[1:], color='blue', label="TTDA")
#plt.plot(range(1,len(medie)), medie_r[1:], color='green', label="TTDA with rotations")
plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*medie[0], color='red', label="Baseline")

plt.xlabel("Number of augmentation")
plt.ylabel("TAG")
plt.title("Task Agnostic with augmentations")
plt.xlim((1,20))
plt.legend()
plt.show()

plt.plot(range(1,len(medie)), medie_TAW[1:], color='blue', label="TTDA")
#plt.plot(range(1,len(medie)), medie_TAW_r[1:], color='green', label="TTDA with rotations")
plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*medie_TAW[0], color='red', label="Baseline")

plt.xlabel("Number of augmentation")
plt.ylabel("TAG")
plt.title("Task Aware with augmentations")
plt.xlim((1,20))
plt.legend()
plt.show()

l = len(medie)-1

print("no augmentation")
print(f"{medie[0]:.1f} ± {std[0]:.1f}")
print(f"{medie_TAW[0]:.1f} ± {std_TAW[0]:.1f}")
print('\n')

print(f"{int(l/2)} augmentations")
print(f"{medie[int(l/2)]:.1f} ± {std[int(l/2)]:.1f}")
#print(f"{medie_TAG[int(l/2)]:.1f} ± {std_TAG[int(l/2)]:.1f}")
print('\n')

print(f"{l} augmentations")
print(f"{medie[l]:.1f} ± {std[l]:.1f}")
#print(f"{medie_TAG[l]:.1f} ± {std_TAG[l]:.1f}")'''

