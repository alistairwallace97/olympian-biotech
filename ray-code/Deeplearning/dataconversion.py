# loading libraries
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, freqz
import operator
import math
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def peakdetection(dataset, sensor, mode):

    MA=[]
    MA = dataset[dataset.columns[sensor]].rolling(window=150).mean()
    #print(MA)
    sensorname = ['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    listpos = 0
    NaNcount = 0
    #print(dataset)
    for datapoint in range(0,len(MA)):   #eliminating NaN if NaN, rollingmean=original data value
        rollingmean = MA[listpos] #Get local mean
        #print(rollingmean)
        if math.isnan(rollingmean) ==1: 
            MA[listpos]=dataset[dataset.columns[sensor]][listpos]

            NaNcount += 1
        listpos += 1
    a=0.02  #set coefficients for different sensors
    b=1.1
    if (sensor == 0) or (sensor == 1):
        a=0.02
        b=1.1
    if (sensor == 4) or (sensor == 5) or (sensor == 6) or (sensor == 7) or (sensor == 8) or (sensor == 9):
        a=0.05
        b=1.01
    if (mode == 1):
        a=0
        b=1

    MA = (MA+dataset[dataset.columns[sensor]].max()*a)*b
    window = []
    peaklist = []
    listpos = 0


    for datapoint in dataset[dataset.columns[sensor]]:
        rollingmean = MA[listpos] #Get local mean
        if (listpos > NaNcount):
        
            if (datapoint > rollingmean): #If signal comes above local mean, mark ROI
                window.append(datapoint)
           
            elif (datapoint < rollingmean) and (len(window) >= 1): #If no detectable R-complex activity -> do nothing
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
                peaklist.append(beatposition) #Add detected peak to list
                window = [] #Clear marked ROI
        listpos += 1  
    return peaklist

def peakcount(list, threshold1, threshold2):#measuring number of peaks in between two thresholds
    count = 0
    for i in range (0,len(list)):
        if list[i]>threshold1 and list[i]<threshold2:
            count+=1
    return count

def statedetection(list,threshold): #detect sitting or moving
    sub=0
    sum=0
    moving=0
    for n in range(0, 6):  
        sub=list[n][1]-list[n][0]
        sum=sum+sub
        #if sub<threshold and list[n][1]!=0 and list[n][0]!=0:
        #    moving = 1
    if sum<30 and list[n][1]!=0 and list[n][0]!=0:
        moving=1

    #print(sum)
    return moving
def motioncorrection(list):#correct misclassified motion
    for i in range (3,len(list)-2):
        if list[i][1] == 1 and list[i-1][1]==0 and list[i-2][1]==0 and list[i+1][1]==0 and list[i+2][1]==0:
            list[i][1]=0
        elif list[i][1] == 0 and list[i-1][1]==1 and list[i-2][1]==1 and list[i+1][1]==1 and list[i+2][1]==1:
            list[i][1]=1
        elif list[i][1] == 1 and list[i-1][1]==0 and list[i-2][1]==0 and list[i-3][1]==0 and list[i+1][1]==0:
            list[i][1]=0
        elif list[i][1] == 0 and list[i-1][1]==1 and list[i-2][1]==1 and list[i-3][1]==1 and list[i+1][1]==1:
            list[i][1]=1
    return list

def motiondetect(df,motionth):#compare the number of peaks of two moving averages to detect motion
    #print(df)
    motionlist = []#list of motions for all df elements
    motion = []#list of pairs of motion and element for every 100 elements
    #number of intersection with MA (a=0.01, b=1.05) used for identifying motion
    peakAx = peakdetection(df, 4, 0) 
    peakAy = peakdetection(df, 5, 0) 
    peakAz = peakdetection(df, 6, 0) 
    peakGx = peakdetection(df, 7, 0) 
    peakGy = peakdetection(df, 8, 0) 
    peakGz = peakdetection(df, 9, 0) 
    #number of intersection with MA (a=0, b=1) used for identifying motion
    crossingAx = peakdetection(df, 4, 1) 
    crossingAy = peakdetection(df, 5, 1) 
    crossingAz = peakdetection(df, 6, 1) 
    crossingGx = peakdetection(df, 7, 1) 
    crossingGy = peakdetection(df, 8, 1) 
    crossingGz = peakdetection(df, 9, 1) 
    for i in range (0,len(df),100):
        peaklistAcc=[]
        th1 = i
        th2=i+100

        peaklistAcc.append([peakcount(peakAx,th1,th2),peakcount(crossingAx,th1,th2)])
        peaklistAcc.append([peakcount(peakAy,th1,th2),peakcount(crossingAy,th1,th2)])
        peaklistAcc.append([peakcount(peakAz,th1,th2),peakcount(crossingAz,th1,th2)])
        peaklistAcc.append([peakcount(peakGx,th1,th2),peakcount(crossingGx,th1,th2)])
        peaklistAcc.append([peakcount(peakGy,th1,th2),peakcount(crossingGy,th1,th2)])
        peaklistAcc.append([peakcount(peakGz,th1,th2),peakcount(crossingGz,th1,th2)])
        #print(peaklistAcc)

        motion.append([th1,statedetection(peaklistAcc,motionth)])
    #print(len(motion))
    motion = motioncorrection(motion)
    
    for i in range(0,len(df)):
        for j in range(0,len(motion)):
            if i >=motion[j][0] and i<motion[j][0]+100:
                motionlist.append(motion[j][1])
    motionlist.extend((0,0))
    #print(len(motionlist))            
    return motionlist

def lowpassfilter(df):
    for i in range(1, 11):
        butter_lowpass(10, 50, order=7)
        df.iloc[:, i] = butter_lowpass_filter(df.iloc[:, i], 10, 50, order=10)
    return df

def difference(df):
    # create feature matrix X and result vector y
    X = np.array(df[df.columns[1:11]]) 	
    y = np.array(df[df.columns[0]])
    #print(df[df.columns[1:11]])
    #print(df[df.columns[0]]) 	
    (n,m)=X.shape   #get no. of rows
    #print(X)
    #print(df)

    count = 0
    D=[]
    Difflabel=['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for i in range(1, n):
        D.append(list(map(operator.sub, X[i-1,0:10], X[i,0:10]))) #from EMG1 to Gz, calculate difference
        count=count+1
    #Diff.extend(D)
    Diff=np.array(D)
    #print(Diff)

    my_df = pd.DataFrame(Diff)
    #print(my_df)

    my_df.to_csv('difference.txt', index=False, header=False)

    ds = pd.read_csv('difference.txt')
    return ds

seq_len = 100

names = ['CoughState', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'InstantHr', 'AvgHr','People']

df = pd.read_csv('combineddata.txt', header=None, names=names)
df=lowpassfilter(df)
ds=difference(df)
motionth = 0.5 #threshold for identifying motion, difference in number of peaks, <2 is moving
motionlist = motiondetect(ds, motionth)
motionseries = pd.Series(motionlist) 
df['Motion'] = motionseries.values  #add motion list to dataframe, its index is 16

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
    if c=='Motion':
        motionlisttemp=listoflist
    np.savetxt(c+".txt", listoflist, delimiter=",", fmt='%s')
outputlist=[]

for i in range(0,len(outputlisttemp)):
<<<<<<< HEAD
    coughbool=0
    motionbool=0
    classi=0
   # if i==len(outputlisttemp):
   #    const=len(outputlisttemp[len(outputlisttemp)-1])%seq_len
    for j in range(0,seq_len):
        if outputlisttemp[i][j]==1:
            coughbool=1
        if motionlisttemp[i][j]==1:
            motionbool=1
    if coughbool==0 and motionbool==0:
        classi=0
    elif coughbool==1 and motionbool==0:
        classi=1
    elif coughbool==0 and motionbool==1:
        classi=2
    elif coughbool==1 and motionbool==1:
        classi=3
    outputlist.append(classi)
=======
    outputbool=0
    for j in range(0,len(outputlisttemp[i][:])):
        if outputlisttemp[i][j-1:j+1]=='1.':
            outputbool=1
    outputlist.append(outputbool)
>>>>>>> ca16d36c11161642990c4fd0fc6544b0e9522f67
np.savetxt("output.txt", outputlist, delimiter=",", fmt='%s')
print("done")
