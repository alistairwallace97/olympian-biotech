# loading libraries
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, freqz

seq_len = 100

names = ['CoughState', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'InstantHr', 'AvgHr','People']

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

for i in range(3, 11):
    butter_lowpass(10, 50, order=7)
    df.iloc[:, i] = butter_lowpass_filter(df.iloc[:, i], 10, 50, order=10)
for i in range(1, 11):
    df.iloc[:, i]=(df.iloc[:, i] - min(df.iloc[:, i]))/ ( max(df.iloc[:, i]) - min(df.iloc[:, i]))



for c in df.columns:    
    start=0
    end=seq_len
    templist=[]
    listoflist=[]
    for i in range (0,len(df)):
        if i>=start and i<end:
            templist.append(df[c][i])
        if i==end-1:
            start=start+seq_len
            templist=" ".join(str(x) for x in templist)
            listoflist.append(templist)
            templist=[]

        end=start+seq_len
    if c=='CoughState':
        outputlisttemp=listoflist
    np.savetxt(c+".txt", listoflist, delimiter=",", fmt='%s')
outputlist=[]

for i in range(0,len(outputlisttemp)):
    outputbool=0
    for j in range(0,len(outputlisttemp[i][:])):
        if outputlisttemp[i][j-1:j+1]=='1.':
            outputbool=1
    outputlist.append(outputbool)
np.savetxt("output.txt", outputlist, delimiter=",", fmt='%s')
print("done")
