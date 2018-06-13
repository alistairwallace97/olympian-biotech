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
import os
import pickle
from sklearn.externals import joblib
import csv
names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People','Motion']  

seq_len = 20

def csvtodf(c):
    dfr = pd.read_csv(c, header=None, names=names)

    dfr=dfr.dropna(how='any') 
    dfr.to_csv('tmpdata.txt', index=False, header=False)
    df = pd.read_csv('tmpdata.txt', header=None, names=names)
    return df

def difference(df):
    # create feature matrix X and result vector y
    X = np.array(df[df.columns[1:11]]) 	
    _ = np.array(df[df.columns[0]])
    (n,_)=X.shape   #get no. of rows

    count = 0
    D=[]
    Difflabel=['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for i in range(1, n):
        D.append(list(map(operator.sub, X[i-1,0:10], X[i,0:10]))) #from EMG1 to Gz, calculate difference
        count=count+1
    Diff=np.array(D)

    my_df = pd.DataFrame(Diff)

    my_df.to_csv('difference.txt', index=False, header=False)

    ds = pd.read_csv('difference.txt', names=Difflabel)
    return ds

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
        a=0.04
        b=1.04
    if (sensor == 4) or (sensor == 5) or (sensor == 6) or (sensor == 7) or (sensor == 8) or (sensor == 9):
        a=0.05
        b=1.01
    if (sensor == 11) or (sensor == 12):
        a=0.01
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
    if sum<30 and list[n][1]!=0 and list[n][0]!=0:
        moving=1

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
    motionlist = []#list of motions for all df elements
    motion = []#list of pairs of motion and element for every seq_len elements
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
    for i in range (0,len(df),seq_len):
        peaklistAcc=[]
        th1 = i
        th2=i+seq_len

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
            if i >=motion[j][0] and i<motion[j][0]+seq_len:
                motionlist.append(motion[j][1])
    motionlist.extend((0,0))
    return motionlist

def dataconverion(df):    #convert into seq_len in one row
    for c in df.columns:    
        start=0
        end=seq_len
        finish=0
        templist=[]
        listoflist=[]
        for i in range (0,len(df)+1):
            if i>=start and i<end:
                templist.append(df[c][i])
            if i==end-1:
                start=start+seq_len
                listoflist.append(templist)
                templist=[]
            if i==round((len(df)/seq_len-1))*seq_len:
                end=start+len(df)%seq_len-1
                finish=1
            elif finish==0:
                end=start+seq_len
        if c=='Cough state':
            outputlisttemp=listoflist
        np.savetxt(c+".csv", listoflist, delimiter=",", fmt='%s')
    outputlist=[]
    for i in range(0,len(outputlisttemp)):
        outputbool=0
        const=seq_len
        if i==len(outputlisttemp)-1:
            const=len(outputlisttemp[len(outputlisttemp)-1])%seq_len
        for j in range(0,const):
            if outputlisttemp[i][j]==1:
                outputbool=1
        outputlist.append(outputbool)
    np.savetxt("output.csv", outputlist, delimiter=",", fmt='%s')

def splitseq_len(peaklist):
    templist=[]
    for i in range (0,len(peaklist)):
        start=peaklist[i]//seq_len*seq_len
        end=start+seq_len
        templist.append([start,end])

    templist=sorted(templist)
    templist=[templist[i] for i in range(len(templist)) if i == 0 or templist[i] != templist[i-1]] # sort and remove duplicates
    return templist
    
def featureextraction(df,templist):   
    featurelist=[]
    featurelisttemp=[]
    ds=difference(df)

    for i in range(0,len(templist)):
        featurelisttemp=[]
        moving=0
        indexbool=0
        for c in df.columns:
            if c!='Cough state' and c!='Hr' and c!='Instant Hr' and c!= 'Avg Hr' and c!= 'People' and c!='Motion' and c!='Index':
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

        featurelist.append(featurelisttemp) #input to machine learning algo

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
    temp =  list(map(operator.sub, Y_pred, Y_validation))
    for i in range(0,len(temp)):
        if temp[i]!=0:
            temp[i]=1
    accuracy=1-np.mean(temp)
    return accuracy

def exportresult(roiaccuracy, coughaccuracy, ypred, y_test, model, knn_n):
    roiaccuracy=roiaccuracy.tolist()
    roiaccuracylist=[roiaccuracy]
    coughaccuracylist=[coughaccuracy]
    y_test.insert(0, "y_test: ")
    roiaccuracylist.insert(0, "ROI accuracy: ")
    coughaccuracylist.insert(0, "Coughs correctly identified: ")

    roiaccuracy=str(roiaccuracy)
    coughaccuracy=str(coughaccuracy)
    result=roiaccuracy+","+coughaccuracy+" \r\n"
    resultname ="mlresult.txt"
    knn_nstr=str(knn_n)
    if model == "knn":
        resultname = model+"_"+knn_nstr+".txt"
    else:
        resultname = model+".txt"
    with open(resultname, "a") as f:
        f.write(result)

def hrcorrection(peaklist):
    notpeak=[]
    for i in range (1,len(peaklist)):
        if peaklist[i]-peaklist[i-1]<10:
            notpeak.append(peaklist[i])

    for i in range (0,len(notpeak)):
        peaklist.remove(notpeak[i])

    return peaklist

def diffHr(peaklist):
    diffpeak=[]
    for i in range (1,len(peaklist)):
        diffpeak.append(peaklist[i]-peaklist[i-1])
    return diffpeak

def calcHr(diffpeak):
    sum=0
    count=0
    Hr=[]
    for i in range (0,len(diffpeak)):
        Hr.append(60/(diffpeak[i]*0.032)) #fs=50Hz
    return Hr
def sleep_detection(df):
    (n,_) = df.shape 
    asleep_list = []
    period = 3000                   #One minute
    if(n > period):
        std_dev_list = []
        threshold = 65
        for i in range(period):
            std_dev_list.append(45.0)
            asleep_list.append(0)
        for i in range(period,n):
            std_dev_list.append(np.std(df["Hr"][i-period+1:i+1]))
            if(std_dev_list[i] < threshold):
                # && df["InstantHr"][i]<df["AvgHr"][i]
                # potentially add this in as if you are 
                # sleeping you Hr should also be pretty low
                asleep_list.append(1)
            else:
                asleep_list.append(0)  
        std_mean = np.mean(std_dev_list[period:])
        for i in range(period):
            std_dev_list[i] = std_mean
        # left in for working out a better threshold
        # value for if someone is sleeping or not
    else:
        asleep_list = [0]*n
    return pd.Series(asleep_list)
def main():
    df_test=csvtodf('./server_local_graph/graph_algo_in.txt')
    df_Hr=df_test
    #df_Hr=csvtodf('combineddata_test.txt')

    ds=difference(df_test)


    indexlist=df_test.index.values.tolist()
    indexlisttemp=indexlist
    indexlist=pd.Series(indexlist)
    df_test['Index'] = indexlist.values#putthing into dataframe
    listofzeros = [0] * len(df_test)
    df_test['EMG1']=listofzeros

    peaklist=indexlist
    peaklist=list(set(peaklist)) #region of interest, points of high differentials
    indexlist=df_test.index.values.tolist()
    indexlisttemp=indexlist
    indexlist=pd.Series(indexlist)
    df_test['Index'] = indexlist.values#putthing into dataframe
    fulllist=splitseq_len(indexlisttemp)
    templist=splitseq_len(peaklist)#list of list, range of start and end of region of interest
    X_test=featureextraction(df_test,templist)#obtain X, 52 columns(5 features for each sensor and 1 for motion, 1 for index)
    y_test=createoutputlist(df_test,templist)

    X_test=featureextraction(df_test,templist)#obtain X, 52 columns(5 features for each sensor and 1 for motion, 1 for index)
    y_test=createoutputlist(df_test,templist)

    X_testtemp=X_test
    y_testtemp=[]
    testindex=X_test['Index'].tolist()
    X_test = X_test.iloc[:,0:51]

    # load the model from disk
    loaded_model = joblib.load('finalized_model.sav')
    ypred=loaded_model.predict(X_test)
        
    print("y_test: ",  *y_test)
    print("y_pred: ",  *ypred)
    print("Index: ", *testindex)
    roiaccuracy=loaded_model.score(X_test, y_test)

    sum=0
    correct=0

    print("Accuracy:  %.6f" % roiaccuracy)

    testconsecutiveindex=[] #list of index of cough signals overlapping two data groups
    predconsecutiveindex=[]

    testcoughcount=0
    for i in range (0,len(y_test)):
        if y_test[i]==1:
            testcoughcount+=1
        if i>0 and y_test[i]==1 and y_test[i-1] == y_test[i]:
            if df_test['Cough state'][i*seq_len]==1 and df_test['Cough state'][i*seq_len-1]==1:
                testconsecutiveindex.append(i)
    testcoughcount=testcoughcount-len(testconsecutiveindex)
    print("Number of coughs in test data: ",testcoughcount)

    predcoughcount=0
    for i in range (0,len(ypred)):
        if ypred[i]==1:
            predcoughcount+=1
    for i in range (0,len(testconsecutiveindex)):
        if ypred[testconsecutiveindex[i]]==1 and ypred[testconsecutiveindex[i]-1]==1:
            predcoughcount-=1
    i=1
    while i<len(ypred):
        if ypred[i]==1 and  ypred[i-1]==1:
            predcoughcount-=1
            i+=1
        i+=1
    print("Number of coughs identified: ",predcoughcount)


    #Hr
    df_Hr=df_Hr.dropna(how='any') 
    df_Hr = lowpassfilter(df_Hr)
    df_Hr['Index'] = indexlist.values#putthing into dataframe
    listofzeros = [0] * len(df_Hr)
    ds=difference(df_Hr)

    #sensor index: 1:EMG1, 2:EMG2, 3:Vibration1, 4:Vibration2, 5:Ax, 6:Ay, 7:Az, 8:Gx, 9:Gy, 10:Gz , 11:Hr1, 12:Hr2, 13:Temperature
    #peaklist1 = peakdetection(df_test, 11, 0)
    peaklist2 = peakdetection(df_test, 12, 0)
    if len(peaklist2)!=0:

        peaklist2=hrcorrection(peaklist2)

        
        #differenceHr1=diffHr(peaklist1)
        differenceHr2=diffHr(peaklist2)
        #differenceHr=[]
        #for i in range(0,len(differenceHr1)):
        #    differenceHr.append((differenceHr1[i]+differenceHr2[i])/2)

        Hr = calcHr(differenceHr2)

        sum=0
        count=0
        for i in range (0,len(Hr)):
            sum=sum+Hr[i]
            count+=1
        if count == 0:
            avg=1
        else:
            avg=sum/count

        Hrlist=[]
        for i in range(0,len(differenceHr2)):
            for j in range(0,round(differenceHr2[i])):
                Hrlist.append(Hr[i])

        df_test=df_test.iloc[0:len(Hrlist), :]
        df_test['Hr']=Hrlist

        #see if sleeping
        sleep_series = sleep_detection(df_test)
        df_test['Sleeping'] = sleep_series.values
        pd.set_option('display.max_rows', 2000)



        #producing three columns for generating graph
        open("./server_local_graph/graph_test.txt", "w").close()

        f= open("./server_local_graph/graph_test.txt", "a")
        f.write("n"+str(predcoughcount)+","+str(avg)+"\n")
        for i in range (0,len(df_test)):
            graphinput=str(df_test['EMG2'][i])+","+str(df_test['Cough state'][i])+","+str(ypred[i//seq_len])+","+str(df_test['Motion'][i])+","+str(df_test['Sleeping'][i])+"\n"
            f.write(graphinput)
        f.close()
    
    else:
                #producing three columns for generating graph
        open("./server_local_graph/graph_test.txt", "w").close()

        f= open("./server_local_graph/graph_test.txt", "a")
        f.write("n"+str(predcoughcount)+","+0+"\n")

        for i in range (0,len(df_test)):
            graphinput=str(df_test['EMG2'][i])+","+str(df_test['Cough state'][i])+","+str(ypred[i//(seq_len)])+","+str(df_test['Motion'][i])+","+"0"+"\n"
            f.write(graphinput)
        f.close()

if __name__ == '__main__':
    main()
