# loading libraries
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, freqz,filtfilt
import operator
import math
import matplotlib.pyplot as plt

seq_len = 100

def csvtodf(c):
    dfr = pd.read_csv(c, header=None, names=names) 
    dfr=dfr.dropna(how='any') 
    dfr.to_csv('tmpdata.txt', index=False, header=False)
    df = pd.read_csv('tmpdata.txt', header=None, names=names)
    return df

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpassfilter(df):  #after this part of dataframe becomes to nan
    for i in range(1, 11):
        df.iloc[:, i] = butter_lowpass_filter(df.iloc[:, i], 10, 50, 10)
    #print(df)
    return df

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #y = filtfilt(b, a, data)
    y = lfilter(b, a, data) #nan here when negative values are passed into this filter
    print(y)
    return y

names = ['CoughState', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'InstantHr', 'AvgHr','People','Motion']

#df = pd.read_csv('combineddata.txt', header=None, names=names)
df = csvtodf('combineddata_test.txt')
#df=lowpassfilter(df)


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
    coughbool=0
    classi=0
   # if i==len(outputlisttemp):
   #    const=len(outputlisttemp[len(outputlisttemp)-1])%seq_len
    for j in range(0,seq_len):
        if outputlisttemp[i][j]=='1':
            coughbool=1
    if coughbool==0:
        classi=0
    elif coughbool==1:
        classi=1
    outputlist.append(classi)
np.savetxt("output.txt", outputlist, delimiter=",", fmt='%s')
print("done!")
