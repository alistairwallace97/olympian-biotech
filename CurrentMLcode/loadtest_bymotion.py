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
names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr1', 'Hr2', 'Temperature','People','Motion']  

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
    #detects peaks and return a list of x values of all detected peaks
    MA=[]

    MA=[]
    MA = dataset[dataset.columns[sensor]].rolling(window=150).mean()
    sensorname = ['EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    listpos = 0
    NaNcount = 0
    for datapoint in range(0,len(MA)):   
        #eliminating NaN if NaN, rollingmean=original data value
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
        #Get local mean
        rollingmean = MA[listpos] 
        if (listpos > NaNcount):
            #If signal comes above local mean, mark ROI
            if (datapoint > rollingmean): 
                window.append(datapoint)
            #If no detectable R-complex activity -> do nothing
            elif (datapoint < rollingmean) and (len(window) >= 1): 
                maximum = max(window)
                #Notate the position of the point on the X-axis
                beatposition = listpos - len(window) + (window.index(max(window))) 
                #Add detected peak to list
                peaklist.append(beatposition) 
                #Clear marked ROI
                window = [] 
        listpos += 1  
    #plot function for debugging purpose
    #if sensor == 2 or sensor == 3:
    #    #Get the y-value of all peaks for plotting purposes
    #    y = [dataset[dataset.columns[sensor]][x] for x in peaklist] 
    #    plt.title("Detected peaks in signal")
    #    plt.xlim(0,len(dataset))
    #    plt.plot(dataset[dataset.columns[sensor]], alpha=0.5, color='blue') 
    #    #Plot moving average
    #    plt.plot(MA, color ='green') 
    #    #Plot detected peaks
    #    plt.scatter(peaklist, y, color='red') 
    #    yy = np.array(df['Cough state']) 	
    #    plt.plot(yy, alpha=0.5, color='green') 
    #    plt.show()
    return peaklist

def peakcombine(list1, list2):
    combined=[]
    for i in range (0,len(list1)):
        for j in range (0,len(list2)):
            if abs(list1[i]-list2[j])<10:
                combined.append(round((list1[i]+list2[j])/2))
    return combined

def peakcount(list, threshold1, threshold2):
    #measuring number of peaks in between two thresholds
    count = 0
    for i in range (0,len(list)):
        if list[i]>threshold1 and list[i]<threshold2:
            count+=1
    return count
    
def splitseq_len(peaklist):
    templist=[]
    for i in range (0,len(peaklist)):
        start=peaklist[i]//seq_len*seq_len
        end=start+seq_len
        templist.append([start,end])

    templist=sorted(templist)
    # sort and remove duplicates
    templist=[templist[i] for i in range(len(templist)) if i == 0 or templist[i] != templist[i-1]] 
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
    #correct misdetected peaks
    notpeak=[]
    for i in range (1,len(peaklist)):
        if peaklist[i]-peaklist[i-1]<10:
            notpeak.append(peaklist[i])

    for i in range (0,len(notpeak)):
        peaklist.remove(notpeak[i])

    return peaklist

def diffHr(peaklist):
    #return list of Diffeence
    diffpeak=[]
    for i in range (1,len(peaklist)):
        diffpeak.append(peaklist[i]-peaklist[i-1])
    return diffpeak

def calcHr(diffpeak):
    sum=0
    count=0
    Hr=[]
    for i in range (0,len(diffpeak)):
        Hr.append(60/(diffpeak[i]*0.032))
    return Hr
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
    
def sleep_detection(df):
    (n,_) = df.shape 
    asleep_list = []
    period = 3000               #One minute
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
    #df_test=csvtodf('combineddata_test.txt')
    df_Hr=df_test

    ds=difference(df_test)


    indexlist=df_test.index.values.tolist()
    indexlisttemp=indexlist
    indexlist=pd.Series(indexlist)

    #putthing into dataframe
    df_test['Index'] = indexlist.values


    indexlist=df_test.index.values.tolist()
    indexlisttemp=indexlist
    indexlist=pd.Series(indexlist)
    df_test['Index'] = indexlist.values

    #filter out moving part
    df_test_still=df_test[df_test['Motion'] == 0]
    df_test_still=df_test_still.reset_index()
    del df_test_still['index']

    #filter out still part
    df_test_moving=df_test[df_test['Motion'] == 1]
    df_test_moving=df_test_moving.reset_index()
    del df_test_moving['index']

    indexlist_still=df_test_still['Index']
    indexlist_moving=df_test_moving['Index']
    #obtaining index

    templist=split(indexlist)
    templist_still=split(indexlist_still)
    templist_moving=split(indexlist_moving)
    #list of list, start and end of seq_len

    X_test=featureextraction(df_test,templist)
    testindex=X_test['Index'].tolist()

    X_test_still=X_test[X_test['Motion'] == 0]
    X_test_moving=X_test[X_test['Motion'] == 1]

    #obtain y for still and moving
    y_test=createoutputlist(df_test,templist)

    # load the model from disk
    loaded_model_still = joblib.load('finalized_model_still.sav')
    loaded_model_moving = joblib.load('finalized_model_moving.sav')

    ypred_still=loaded_model_still.predict(X_test_still)
    ypred_moving=loaded_model_moving.predict(X_test_moving)
    

    ypred=[]
    templist_still_pos=0
    templist_moving_pos=0

    for i in range (0,len(templist)):
        if templist[i]==templist_still[templist_still_pos]:
            ypred.append(ypred_still[templist_still_pos])
            templist_still_pos+=1
        else:
            ypred.append(ypred_moving[templist_moving_pos])
            templist_moving_pos+=1

    print(ypred)

    print("y_test: ",  *y_test)
    print("y_pred: ",  *ypred)
    print("Index: ", *testindex)
    Accuracy=accuracy(y_test, ypred)


    print("Accuracy:  %.6f" % Accuracy)

    testconsecutiveindex=[] 
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

    i=1
    while i<len(ypred):
        if ypred[i]==1 and  ypred[i-1]==1:
            predcoughcount-=1
            i+=1
        i+=1
    print("Number of coughs identified: ",predcoughcount)


    #Hr
    df_Hr=df_Hr.dropna(how='any') 
    df_Hr['Index'] = indexlist.values#putthing into dataframe
    listofzeros = [0] * len(df_Hr)
    ds=difference(df_Hr)
    #sensor index: 1:EMG1, 2:EMG2, 3:Vibration1, 4:Vibration2, 5:Ax, 6:Ay, 7:Az, 8:Gx, 9:Gy, 10:Gz , 11:Hr1, 12:Hr2, 13:Temperature
    peaklist1 = peakdetection(df_test, 11, 0)
    peaklist2 = peakdetection(df_test, 12, 0)
    if len(peaklist1)!=0 or len(peaklist2)!=0:
        peaklist1=hrcorrection(peaklist1)
        peaklist2=hrcorrection(peaklist2)
        
        differenceHr1=diffHr(peaklist1)
        differenceHr2=diffHr(peaklist2)
        #Choose a better Hr
        if len(peaklist1)!=0 > len(peaklist2)!=0:
            differenceHr=differenceHr1
        else:
            differenceHr=differenceHr2
        Hr = calcHr(differenceHr)
        sum=0
        count=0
        for i in range (0,len(Hr)):
            sum=sum+Hr[i]
            count+=1
        if count == 0:
            avg=1
        else:
            avg=sum/count

        #create a Hr list to add it to the dataframe
        Hrlist=[]
        for i in range (0,peaklist2[0]):
            Hrlist.append(Hr[0])

        start=peaklist2[0]
        for i in range(0,len(Hr)):
            end=peaklist2[i+1]
            for j in range(start,end):
                Hrlist.append(Hr[i])
            start=end

        for i in range (peaklist2[-1],len(df_test)):
            Hrlist.append(Hr[-1])

        df_test['Hr']=Hrlist

        #see if sleeping
        sleep_series = sleep_detection(df_test)
        df_test['Sleeping'] = sleep_series.values

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
        f.write("n"+str(predcoughcount)+","+"0"+"\n")

        for i in range (0,len(df_test)):
            graphinput=str(df_test['EMG2'][i])+","+str(df_test['Cough state'][i])+","+str(ypred[i//(seq_len)])+","+str(df_test['Motion'][i])+","+"0"+"\n"
            f.write(graphinput)
        f.close()

if __name__ == '__main__':
    main()
