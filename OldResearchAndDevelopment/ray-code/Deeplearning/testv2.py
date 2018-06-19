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
from sklearn.tree import DecisionTreeClassifier  

def csvtodf(c):
    dfr = pd.read_csv(c, header=None, names=names)

    #dfr.drop(dfr.index[:1],inplace = True) 
    dfr=dfr.dropna(how='any') 
    dfr.to_csv('tmpdata.txt', index=False, header=False)
    df = pd.read_csv('tmpdata.txt', header=None, names=names)
    df=abs(df)
    #print(df.head())
    #print(df)
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
   
    #print(sensor)
    #print(peaklist)
    #y = [dataset[dataset.columns[sensor]][x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
    #plt.title("Detected peaks in signal")
    #plt.xlim(0,10126)
    #plt.plot(dataset[dataset.columns[sensor]], alpha=0.5, color='blue') #Plot semi-transparent HR
    #plt.plot(MA, color ='green') #Plot moving average
    #plt.scatter(peaklist, y, color='red') #Plot detected peaks
    #plt.show()
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

def peakcount(list, threshold1, threshold2):#measuring number of peaks in between two thresholds
    count = 0
    for i in range (0,len(list)):
        if list[i]>threshold1 and list[i]<threshold2:
            count+=1
    return count
    
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

def dataconverion(df):    #convert into 100 in one row
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
        np.savetxt(c+".csv", listoflist, delimiter=",", fmt='%s')
    outputlist=[]
    for i in range(0,len(outputlisttemp)):
        outputbool=0
        const=100
        if i==len(outputlisttemp)-1:
            const=len(outputlisttemp[len(outputlisttemp)-1])%100
        for j in range(0,const):
            if outputlisttemp[i][j]==1:
                outputbool=1
        outputlist.append(outputbool)
    #print(outputlist)
    np.savetxt("output.csv", outputlist, delimiter=",", fmt='%s')

def split100(peaklist):
    #print(peaklist)
    templist=[]
    for i in range (0,len(peaklist)):
        start=peaklist[i]//100*100
        end=start+100
        templist.append([start,end])

    templist=sorted(templist)
    templist=[templist[i] for i in range(len(templist)) if i == 0 or templist[i] != templist[i-1]] # sort and remove duplicates
    #print(templist) #list of list, start and end of region of interest
    #print(len(templist))
    return templist
    
def featureextraction(df,templist):   
    featurelist=[]
    featurelisttemp=[]

    for i in range(0,len(templist)):
        featurelisttemp=[]
        moving=0
        #print(df.tail())
        for c in df.columns:
            if c!='Cough state' and c!='Hr' and c!='Instant Hr' and c!= 'Avg Hr' and c!= 'People' and c!='Motion':
                data=df[c][templist[i][0]:templist[i][1]]
                datamax=data.max()
                datamin=data.min()
                datamean=data.mean()
                datavar=data.var()
                #diff=ds[c][100:200]
                #maxdiff=abs(diff).max()            
                featurelisttemp.append(datamax)
                featurelisttemp.append(datamin)
                featurelisttemp.append(datamean)
                featurelisttemp.append(datavar)
            elif c=='Motion':
                for j in range (templist[i][0],templist[i][1]):
                    if j<len(df):
                        if df[c][j]==1:
                            moving=1
                featurelisttemp.append(moving)
                moving=0


        featurelist.append(featurelisttemp) #input to machine learning algo
    #print(len(featurelist))

    #creating labels
    sensorlabels=['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    featurenames=['Max','Min','Mean','Var']
    fulllabel=[]
    for i in range (0,len(sensorlabels)):
        for j in range (0,len(featurenames)):
            fulllabel.append(sensorlabels[i]+featurenames[j])
    fulllabel.append('Motion')
    X=pd.DataFrame(featurelist,columns = fulllabel)
    #print(X)
    return X

def createoutputlist(df,templist):
    #create output list
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
    temp = Y_pred - Y_validation
    for i in range(0,len(temp)):
        if temp[i]!=0:
            temp[i]=1
    accuracy=1-np.mean(temp)
    return accuracy

# define column names
names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People']

df=csvtodf('combineddata.txt')
#combineddata.txt
#halfdata.txt
#Ismaeel_test2.txt
#Ali_test3

#low pass filter
df=lowpassfilter(df)
ds=difference(df)
#sensor index: 0:EMG1, 1:EMG2, 2:Vibration1, 3:Vibration2, 4:Ax, 5:Ay, 6:Az, 7:Gx, 8:Gy, 9:Gz 
#peak detection using moving avg
motionth = 0.5 #threshold for identifying motion, difference in number of peaks, <2 is moving
motionlist = motiondetect(ds, motionth)
motionseries = pd.Series(motionlist)
df['Motion'] = motionseries.values
#print(df.head())
peaklist1 = peakdetection(ds, 0, 0)
peaklist2 = peakdetection(ds, 1, 0)
peaklist = peakcombine(peaklist1,peaklist2) #put common elements into a set
peaklist=list(set(peaklist)) #region of interest, points of high differentials

print(peaklist)
#plot region of interest
y = [df.EMG1[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
plt.title("Detected peaks in signal")
plt.xlim(0,10126)
plt.plot(df.EMG1, alpha=0.5, color='blue')
plt.scatter(peaklist, y, color='red') #Plot detected peaks
yy = np.array(df['CoughState']*1000) 	
plt.plot(yy, alpha=0.5, color='green') 
plt.show()

#feature extraction and creating output list
templist=split100(peaklist)
#print(templist)
X=featureextraction(df,templist)
outputlist=createoutputlist(df,templist)

#print(featurelist[0:41])
X_train, X_test, y_train, y_test = train_test_split(X, outputlist, test_size=0.1)

normalize(X_train,X_test)

#print (len(X_train),len(y_train))
#print (len(X_test), len(y_test))
#print (X_train.head())

#print (X_train.iloc[:, 1])
#print (y_train)
#print (X_test)
#print (y_test)

#X_train = X_train.sort_index()
#X_test = X_test.sort_index()
#y_train = y_train.sort_index()
#y_test = y_test.sort_index()

#KNeighborsClassifier(n_neighbors=5)
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train) 
ypred=classifier.predict(X_test)
#ypred=knn.predict(testX)
print(y_test)
print(ypred)
print("Accuracy: %.6f" % accuracy(y_test, ypred))
#print(ypredict)
#print(y_test)
