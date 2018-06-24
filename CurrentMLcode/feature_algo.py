'''
This script reads the preprocessed data, extracts features, feeds extracted features 
into machine learning model to train and exports machine learning model as finalized_model.sav file.
It also reads test data in testdata file and runs the model on it and prints accuracy for checking purpose.
'''

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
from sklearn import datasets, linear_model
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  
import os
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import cross_validate

seq_len = 20

def csvtodf(c): 
    #converts csv to df and drops rows containing NaN
    dfr = pd.read_csv(c, header=None, names=names)
    dfr=dfr.dropna(how='any') 
    dfr.to_csv('tmpdata.txt', index=False, header=False)
    df = pd.read_csv('tmpdata.txt', header=None, names=names)
    return df

def difference(df):
    # create feature matrix X and result vector y
    X = np.array(df[df.columns[1:11]]) 	
    y = np.array(df[df.columns[0]])
	
    (n,m)=X.shape   #get no. of rows

    count = 0
    D=[]
    Difflabel=['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for i in range(1, n):
        #calculate difference
        D.append(list(map(operator.sub, X[i-1,0:10], X[i,0:10]))) 
        count=count+1
    #Diff.extend(D)
    Diff=np.array(D)
    my_df = pd.DataFrame(Diff)
    my_df.to_csv('difference.txt', index=False, header=False)
    ds = pd.read_csv('difference.txt', names=Difflabel)
    return ds

def peakdetection(dataset, sensor, mode):
    #detects peaks and return a list of x values of all detected peaks
    MA=[]
    MA = dataset[dataset.columns[sensor]].rolling(window=150).mean()
    sensorname = ['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    listpos = 0
    NaNcount = 0
    for datapoint in range(0,len(MA)):   
        #eliminating NaN. If NaN, rollingmean=original data value
        rollingmean = MA[listpos] 
        #Get local mean
        if math.isnan(rollingmean) ==1: 
            MA[listpos]=dataset[dataset.columns[sensor]][listpos]

            NaNcount += 1
        listpos += 1
    #set coefficients for different sensors
    a=0.02  
    b=1.1
    if (sensor == 0) or (sensor == 1):
        a=0.03
        b=1.03
    if (sensor == 4) or (sensor == 5) or (sensor == 6):
        a=0.025
        b=1.025
    if (sensor == 7) or (sensor == 8) or (sensor == 9):
        a=0.045
        b=1.045    
    if (mode == 1):
        a=0
        b=1

    MA = (MA+dataset[dataset.columns[sensor]].max()*a)*b
    window = []
    peaklist = []
    listpos = 0

    for datapoint in dataset[dataset.columns[sensor]]:
        #Get local mean
        rollingmean = MA[listpos] 
        if (listpos > NaNcount):
            #If signal comes above local mean, mark ROI
            if (datapoint > rollingmean): 
                window.append(datapoint)
            #If no detectable R-complex activity -> do nothing
            elif (datapoint < rollingmean) and (len(window) >= 1):
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(max(window))) 
                #Notate the position of the point on the X-axis
                peaklist.append(beatposition) 
                #Add detected peak to list
                window = [] 
                #Clear marked ROI
        listpos += 1  
    return peaklist

def lowpassfilter(df):
    for i in range(1, 11):
        butter_lowpass(10, 50, order=7)
        df.iloc[:, i] = butter_lowpass_filter(df.iloc[:, i], 10, 50, order=10)
    return df

def peakcombine(list1, list2):
    combined=[]
    for i in range (0,len(list1)):
        for j in range (0,len(list2)):
            if abs(list1[i]-list2[j])<10:
                combined.append(round((list1[i]+list2[j])/2))
    return combined

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def split(peaklist):
    templist=[]
    for i in range (0,len(peaklist)):
        start=peaklist[i]//seq_len*seq_len
        end=start+seq_len
        templist.append([start,end])

    templist=sorted(templist)
    templist=[templist[i] for i in range(len(templist)) if i == 0 or templist[i] != templist[i-1]] 
    # sort and remove duplicates

    return templist
    
def featureextraction(df,templist):  
    #return a df of features extracted from input df
    featurelist=[]
    featurelisttemp=[]
    ds=difference(df)

    for i in range(0,len(templist)):
        featurelisttemp=[]
        moving=0
        indexbool=0
        for c in df.columns:
            if c!='Cough state' and c!='Hr1' and c!='Hr2' and c!= 'Temperature' and c!= 'People' and c!='Motion' and c!='Index':
                data=df[c][templist[i][0]:templist[i][1]]
                datamax=data.max()
                datamin=data.min()
                datamean=data.mean()
                datavar=data.var()
                diff=ds[c][templist[i][0]:templist[i][1]]
                maxdiff=diff.max()
                mindiff=diff.min()
                         
                featurelisttemp.append(datamax)
                featurelisttemp.append(datamin)
                featurelisttemp.append(datamean)
                featurelisttemp.append(datavar)
                if maxdiff>abs(mindiff):
                    featurelisttemp.append(maxdiff)
                else:
                    featurelisttemp.append(mindiff)



            elif c=='Motion':
                for j in range (templist[i][0],templist[i][1]):
                    if j<len(df):
                        if df[c][j]==1:
                            moving=1
                featurelisttemp.append(moving)
                moving=0
            elif c=='Index':
                for j in range (templist[i][0],templist[i][1]):
                    if j<len(df):
                        indexbool
                featurelisttemp.append(templist[i][0])

        featurelist.append(featurelisttemp) 
        #input to machine learning algo

    #creating labels
    sensorlabels=['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    featurenames=['Max','Min','Mean','Var', 'Max diff']
    fulllabel=[]
    for i in range (0,len(sensorlabels)):
        for j in range (0,len(featurenames)):
            fulllabel.append(sensorlabels[i]+featurenames[j])
    fulllabel.append('Motion')
    fulllabel.append('Index')
    X=pd.DataFrame(featurelist,columns = fulllabel)
    return X

def createoutputlist(df,templist):
    #create output list from cough status column in df
    outputlist=[]
    cough=0
    for i in range(0,len(templist)):
        for j in range(templist[i][0],templist[i][1]):
            if j<len(df):
                if df['Cough state'][j]==1:
                    cough=1
        outputlist.append(cough)
        cough=0
    return outputlist

def normalize(X_train,X_test):
    for i in range(0, 10):
        datamax = max(X_train.iloc[:, i])
        datamin = min(X_train.iloc[:, i])
        X_train.iloc[:, i]=(X_train.iloc[:, i] - datamin)/ (datamax - datamin)
        X_test.iloc[:, i]=(X_test.iloc[:, i] - datamin)/ (datamax - datamin)

def accuracy( Y_validation, Y_pred ):
    temp =  list(map(operator.sub, Y_pred, Y_validation))
    for i in range(0,len(temp)):
        if temp[i]!=0:
            temp[i]=1
    accuracy=1-np.mean(temp)
    return accuracy

# define column names
names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr1', 'Hr2', 'Temperature', 'People', 'Motion']
df=csvtodf('combineddata_train.txt')
ds=difference(df)

indexlist=df.index.values.tolist()
indexlist=pd.Series(indexlist)
df['Index'] = indexlist.values
#putting into dataframe

listofzeros = [0] * len(df)
df['EMG1']=listofzeros

templist=split(indexlist)

X_train=featureextraction(df,templist)
#obtain X, 52 columns(5 features for each sensor and 1 for motion, 1 for index)
X_train = X_train.iloc[:,0:50]
y_train=createoutputlist(df,templist)


df_test=csvtodf('combineddata_test.txt')

ds=difference(df_test)

#sensor index: 0:EMG1, 1:EMG2, 2:Vibration1, 3:Vibration2, 4:Ax, 5:Ay, 6:Az, 7:Gx, 8:Gy, 9:Gz 

indexlist=df_test.index.values.tolist()
indexlisttemp=indexlist
indexlist=pd.Series(indexlist)
df_test['Index'] = indexlist.values#putthing into dataframe

fulllist=split(indexlisttemp)
y_testfull=createoutputlist(df_test,fulllist)

peaklist=indexlist
templist=split(peaklist)#list of list, range of start and end of region of interest
X_test=featureextraction(df_test,templist)#obtain X, 42 columns(5 features for each sensor and 1 for motion, 1 for index)
y_test=createoutputlist(df_test,templist)
#pulling out index and saving it for later reference
X_testtemp=X_test
y_testtemp=[]

testindex=X_test['Index'].tolist()
X_test = X_test.iloc[:,0:50]

#defining ml models
knn_n=5
model="DecisionTree"
#classifier = DecisionTreeClassifier() 
#classifier = KNeighborsClassifier(n_neighbors=knn_n)
#classifier = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.1, max_depth=2, random_state=0)#need to adjust learning rate 
classifier = RandomForestClassifier(n_estimators=1000)
classifier.fit(X_train, y_train) 
ypred=classifier.predict(X_test)

cv_results = cross_validate(classifier, X_train, y_train, return_train_score=False, scoring = ('accuracy'))
print(cv_results)

#print result
print("Result: ")
print("y_test: ",  *y_test)
print("y_pred: ",  *ypred)
print("Index: ", *testindex)
roiaccuracy=accuracy(y_test, ypred)
sum=0
correct=0
coughaccuracy=0
for i in range(0,len(y_test)):
    if y_test[i]==1:
        if ypred[i]==1:
            correct+=1
        sum+=1
if sum==0:
    coughaccuracy=-1
else:
    coughaccuracy=correct/sum
print("Accuracy:  %.6f" % roiaccuracy)
print("Coughs correctly identified:  %.6f" % coughaccuracy)



filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))