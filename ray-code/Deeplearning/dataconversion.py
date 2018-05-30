# loading libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
import operator
import math
from scipy.signal import butter, lfilter, freqz
import string

names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People']

df = pd.read_csv('Ali_test3.txt', header=None, names=names)

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#def outputcount

for i in range(3, 11):
    butter_lowpass(10, 50, order=7)
    df.iloc[:, i] = butter_lowpass_filter(df.iloc[:, i], 10, 50, order=10)
for i in range(1, 11):
    df.iloc[:, i]=(df.iloc[:, i] - min(df.iloc[:, i]))/ ( max(df.iloc[:, i]) - min(df.iloc[:, i]))



#print(listoflist)

print(len(df))
for c in df.columns:    
    start=0
    end=100
    templist=[]
    listoflist=[]
    for i in range (0,len(df)):
        if i>=start and i<end:
            templist.append(df[c][i])
        if i==end-1:
            start=start+100
            templist=" ".join(str(x) for x in templist)
            listoflist.append(templist)
            #print(len(templist))
            templist=[]

        end=start+100
    if c=='Cough state':
        outputlisttemp=listoflist
    #print(listoflist)
    np.savetxt(c+".csv", listoflist, delimiter=",", fmt='%s')
print(len(listoflist))
outputlist=[]

for i in range(0,len(outputlisttemp)):
    outputbool=0
    if i==len(outputlisttemp):
        const=len(outputlisttemp[len(outputlisttemp)-1])%100
    for j in range(0,100):
        if outputlisttemp[i][j]=="1":
            outputbool=1
    outputlist.append(outputbool)
np.savetxt("output.csv", outputlist, delimiter=",", fmt='%s')



#dat=pd.read_csv("EMG1.csv", delimiter=",")
#print(dat)

#to_write = []
#counter = 0
#with open("EMG1.csv", "r") as f:
#    for line in f:
#        for counter in range(0,102):
#            line = list(line)
#            line[line.index(",")] = "/"
#            line[line.index(",")] = "/"
#            counter += 1
#        counter = 0
#        to_write.append("".join(line))

#for c in df.columns:    
 #   data = pd.read_csv(c+".csv", header=None, names=names)
  #  print(data)

##data = pd.read_csv("EMG1.csv", header=None)
#print(data)
#for i in range(0,)