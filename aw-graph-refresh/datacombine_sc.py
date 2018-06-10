# loading libraries
import pandas as pd
import numpy as np
import glob
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import operator
import math
from scipy.signal import butter, lfilter, freqz
from sklearn import datasets, linear_model
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  
import pickle
from sklearn.externals import joblib

def standardization(X_train, Mean, Std):
    for i in range(1, 11):
        X_train.iloc[:, i]=(X_train.iloc[:, i] -Mean[i-1] )/ (Std[i-1])
    return X_train

def peakdetection(dataset, sensor, mode):

    MA=[]
    MA = dataset[dataset.columns[sensor]].rolling(window=150).mean()
    sensorname = ['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    listpos = 0
    NaNcount = 0
    for datapoint in range(0,len(MA)):   #eliminating NaN if NaN, rollingmean=original data value
        rollingmean = MA[listpos] #Get local mean
        if math.isnan(rollingmean) ==1: 
            MA[listpos]=dataset[dataset.columns[sensor]][listpos]

            NaNcount += 1
        listpos += 1
    a=0.02  #set coefficients for different sensors
    b=1.1
    if (sensor == 0) or (sensor == 1):
        a=0.03
        b=1.03
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
    if sensor == 2 or sensor == 3:
        y = [dataset[dataset.columns[sensor]][x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
        plt.title("Detected peaks in signal")
        plt.xlim(0,len(dataset))
        plt.plot(dataset[dataset.columns[sensor]], alpha=0.5, color='blue') #Plot semi-transparent HR
        plt.plot(MA, color ='green') #Plot moving average
        plt.scatter(peaklist, y, color='red') #Plot detected peaks
        yy = np.array(df['Cough state']) 	
        plt.plot(yy, alpha=0.5, color='green') 
        plt.show()
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

    return moving

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def lowpassfilter(df): 
    for i in range(1, 11):
        df.iloc[:, i] = butter_lowpass_filter(df.iloc[:, i], 10, 50, 10)
    return df
def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #y = filtfilt(b, a, data)
    y = lfilter(b, a, data) #nan here when negative values are passed into this filter
    return y
def difference(df):
    # create feature matrix X and result vector y
    X = np.array(df[df.columns[1:11]]) 	
    y = np.array(df[df.columns[0]])	
    (n,m)=X.shape   #get no. of rows

    count = 0
    D=[]
    Difflabel=['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for i in range(1, n):
        D.append(list(map(operator.sub, X[i-1,0:10], X[i,0:10]))) #from EMG1 to Gz, calculate difference
        count=count+1
    #Diff.extend(D)
    Diff=np.array(D)
    #print(Diff)

    my_df = pd.DataFrame(Diff, columns=Difflabel)

    my_df.to_csv('difference.txt', index=False, header=False)

    ds = pd.read_csv('difference.txt', names=Difflabel)
    return ds
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

        motion.append([th1,statedetection(peaklistAcc,motionth)])
    motion = motioncorrection(motion)
    for i in range(0,len(df)):
        for j in range(0,len(motion)):
            if i >=motion[j][0] and i<motion[j][0]+100:
                motionlist.append(motion[j][1])
    motionlist.append((0))
    return motionlist
def motionrange(df):
    motionrangelist=[]
    start=0
    for i in range(1,len(df)):
        if df['Motion'][i]!=df['Motion'][i-1]:
            end=i
            motionrangelist.append([start,end])
            start=end
    return motionrangelist

pathtrain = "./traindata/"
all_filestrain = glob.glob(os.path.join(pathtrain, "*.txt")) #make list of paths
df = pd.DataFrame()

pathtest = "./testdata/"
all_filestest = glob.glob(os.path.join(pathtest, "*.txt")) #make list of paths
dftest = pd.DataFrame()


initials_to_number = {"aw":0.0, "sc":1.0, "lj":2.0,\
                        "ls":3.0, "ir":4.0, "ik":5.0,\
                        "sa":6.0}
names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People']


for file in all_filestrain:
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    if(file_name[0:2] != '00'):        
        # Reading the file content to create a DataFrame
        dftmp = pd.read_csv(file, header=None, names=names, sep=',').convert_objects(convert_numeric=True)
        #drop the first and last line of code which are corrupted by the start/stop action
        dftmp.drop(dftmp.index[:1],inplace = True)
        dftmp.drop(dftmp.tail(1).index,inplace=True)
        #standardization of each person's file
        #dftmp = standardization(dftmp)
        dftmp = lowpassfilter(dftmp)
        dftmp=dftmp.dropna(how='any') 
        dftmp.to_csv('tmpdata.txt', index=False, header=False)
        dftmp = pd.read_csv('tmpdata.txt', header=None, names=names)
        ds=difference(dftmp)

        #peak detection using moving avg
        motionth = 0.5 #threshold for identifying motion, difference in number of peaks, <2 is moving
        motionlist = motiondetect(ds, motionth)
        motionseries = pd.Series(motionlist)
        dftmp['Motion'] = motionseries.values
        #motionrangelist=motionrange(dftmp)
        start=0
        end=100

        # Make a temporary .txt file in csv form so we can
        # look at columns
        dftmp.to_csv('tmp.txt', index=False, header=False)
        dfnum = pd.read_csv('tmp.txt', header=None, names=names)
        #print(len(dftmp)//100*100)
        dftmp=dftmp[:len(dftmp)//100*100]
        #print(motionlist)
        #print(dftmp.iloc[8200])
        # If the person forgot to set the number, then 
        # reset the number for them automatically.
        #print(dfnum)
        if(initials_to_number[file_name[0:2]] != dftmp['People'][14]):
            for i in range(0,len(dftmp)):
                dftmp.loc[i, 'People'] = initials_to_number[file_name[0:2]]

        df = df.append(dftmp)
Mean=[]
Std=[]
for i in range(1, 11):
    Mean.append(df.iloc[:, i].mean())
    Std.append(df.iloc[:, i].std())
df=standardization(df, Mean, Std)
df.to_csv('combineddata_train.txt', index=False, header=False)

for file in all_filestest:
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    if(file_name[0:2] != '00'):        
        # Reading the file content to create a DataFrame
        dftmp = pd.read_csv(file, header=None, names=names, sep=',').convert_objects(convert_numeric=True)
        #drop the first and last line of code which are corrupted by the start/stop action
        dftmp.drop(dftmp.index[:1],inplace = True)
        dftmp.drop(dftmp.tail(1).index,inplace=True)
        #standardization of each person's file
        #dftmp = standardization(dftmp)
        dftmp = lowpassfilter(dftmp)
        dftmp=dftmp.dropna(how='any') 
        dftmp.to_csv('tmpdata.txt', index=False, header=False)
        dftmp = pd.read_csv('tmpdata.txt', header=None, names=names)
        ds=difference(dftmp)

        #peak detection using moving avg
        motionth = 0.5 #threshold for identifying motion, difference in number of peaks, <2 is moving
        motionlist = motiondetect(ds, motionth)
        motionseries = pd.Series(motionlist)
        dftmp['Motion'] = motionseries.values
        #motionrangelist=motionrange(dftmp)
        start=0
        end=100
        
        # Make a temporary .txt file in csv form so we can
        # look at columns
        dftmp.to_csv('tmp.txt', index=False, header=False)
        dfnum = pd.read_csv('tmp.txt', header=None, names=names)
        #print(len(dftmp)//100*100)
        dftmp=dftmp[:len(dftmp)//100*100]
        #print(motionlist)
        #print(dftmp.iloc[8200])
        # If the person forgot to set the number, then 
        # reset the number for them automatically.
        #print(dfnum)
        if(initials_to_number[file_name[0:2]] != dftmp['People'][14]):
            for i in range(0,len(dftmp)):
                dftmp.loc[i, 'People'] = initials_to_number[file_name[0:2]]

        dftest = dftest.append(dftmp)
dftest=standardization(dftest, Mean, Std)
dftest.to_csv('combineddata_test.txt', index=False, header=False)

# Delete temporary .txt file for reseting the person number
#os.remove("tmp.txt")

print("done")