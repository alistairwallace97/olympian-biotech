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

names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People']

df = pd.read_csv('combineddata.txt', header=None, names=names)

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
#Filtered signal generates nan for some reason, comment out at the moment
#for i in range(3, 11):
#    butter_lowpass(10, 50, order=7)
#    df.iloc[:, i] = butter_lowpass_filter(df.iloc[:, i], 10, 50, order=10)
for i in range(1, 11):
    df.iloc[:, i]=(df.iloc[:, i] - min(df.iloc[:, i]))/ ( max(df.iloc[:, i]) - min(df.iloc[:, i]))


#print(listoflist)

for c in df.columns:    
    start=0
    end=100
    finish=0
    templist=[]
    listoflist=[]
    for i in range (0,len(df)+1):
        if i>=start and i<end:
            templist.append(df[c][i])
        if i==end-1:
            start=start+100
            templist = " ".join(str(x)for x in templist)
            listoflist.append(templist)
            #print(len(templist))
            templist=[]
        if i==round((len(df)/100-1))*100:
            end=start+len(df)%100-1
            #print(end)
            finish=1
        elif finish==0:
            end=start+100
    if c=='Cough state':
        outputlisttemp=listoflist
    np.savetxt(c+".txt", listoflist, delimiter=" ", fmt='%s')
outputlist=[]
print(len(outputlisttemp[len(outputlisttemp)-1])%100-1)
for i in range(0,len(outputlisttemp)):
    outputbool=0
    const=100
    if i==len(outputlisttemp)-1:
        const=len(outputlisttemp[len(outputlisttemp)-1])%100
    for j in range(0,const):
        if outputlisttemp[i][j]==1:
            outputbool=1
    outputlist.append(outputbool)
np.savetxt ("output.txt", outputlist, delimiter=" ", fmt='%s')
print("done!")

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