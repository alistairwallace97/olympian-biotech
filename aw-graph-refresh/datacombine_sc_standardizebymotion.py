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

seq_len = 20

def standardization(X_train, Mean, Std):
    for i in range(1, 11):
        X_train.iloc[:, i]=(X_train.iloc[:, i] -Mean[i-1] )/ (Std[i-1])
    return X_train

def peakdetection(dataset, sensor, mode):
    #detects peaks and return a list of x values of all detected peaks
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
        a=0.03
        b=1.03
    if (sensor == 4) or (sensor == 5) or (sensor == 6):
        a=0.05
        b=1.01
    if (sensor == 7) or (sensor == 8) or (sensor == 9):
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
    #if sensor == 4 or sensor == 5 or sensor==6: #plot function for debugging purpose
    #    y = [dataset[dataset.columns[sensor]][x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
    #    plt.title("Detected peaks in signal")
    #    plt.xlim(0,len(dataset))
    #    plt.plot(dataset[dataset.columns[sensor]], alpha=0.5, color='blue') #Plot semi-transparent HR
    #    plt.plot(MA, color ='green') #Plot moving average
    #    plt.scatter(peaklist, y, color='red') #Plot detected peaks
    #    plt.show()
    return peaklist

def peakcount(list, threshold1, threshold2):
    #measuring number of peaks in between two thresholds
    count = 0
    for i in range (0,len(list)):
        if list[i]>threshold1 and list[i]<threshold2:
            count+=1
    return count

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
    y = lfilter(b, a, data)
    return y

def difference(df): 
    #returns df of differences
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
    Diff=np.array(D)
    my_df = pd.DataFrame(Diff, columns=Difflabel)
    my_df.to_csv('difference.txt', index=False, header=False)
    ds = pd.read_csv('difference.txt', names=Difflabel)
    return ds

def motioncorrection(list):
    #correct misclassified motion
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

def statedetection(list,threshold): 
    #detect sitting or moving
    sub=0
    sum=0
    moving=0
    for n in range(0, 6):  
        sub=list[n][1]-list[n][0]
        sum=sum+sub
    if sum<threshold and list[n][1]!=0 and list[n][0]!=0:
        moving=1
    return moving

def motiondetect(df,motionth):
    #compare the number of peaks of two moving averages to detect motion
    motionlist = []
    motion = []
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
    motionlist.append((0))
    return motionlist

def dataconverion(df):    
    #convert and export txt that has seq_len elements per row for deeplearning input
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

def motionrange(df): 
    #create and returns list of list of ranges that have same motion 
    motionrangelist=[]
    start=0
    for i in range(1,len(df)):
        if df['Motion'][i]!=df['Motion'][i-1]:
            end=i
            motionrangelist.append([start,end])
            start=end
    return motionrangelist

def main(mode):
    initials_to_number = {"aw":0.0, "sc":1.0, "lj":2.0,\
                            "ls":3.0, "ir":4.0, "ik":5.0,\
                            "sa":6.0, "te":7.0}
    names = ['CoughState', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr1', 'Hr2', 'Temperature','People']

    if((mode == 'train') or (mode == 'both')):
        pathtrain = "./traindata/"
        all_filestrain = glob.glob(os.path.join(pathtrain, "*.txt")) #make list of paths
        df = pd.DataFrame()

        for file in all_filestrain:
            # Getting the file name without extension
            file_name = os.path.splitext(os.path.basename(file))[0]
            if(file_name[0:2] != '00'):        
                # Reading the file content to create a DataFrame
                dftmp = pd.read_csv(file, header=None, names=names, sep=',').convert_objects(convert_numeric=True)
                #drop the first and last line of code which are corrupted by the start/stop action
                dftmp.drop(dftmp.index[:1],inplace = True)
                dftmp.drop(dftmp.tail(1).index,inplace=True)

                dftmp = lowpassfilter(dftmp)
                dftmp=dftmp.dropna(how='any') 
                dftmp.to_csv('tmpdata.txt', index=False, header=False)
                dftmp = pd.read_csv('tmpdata.txt', header=None, names=names)
                ds=difference(dftmp)

                motionth = 10 #threshold for identifying motion, difference in number of peaks
                motionlist = motiondetect(ds, motionth)
                motionseries = pd.Series(motionlist)
                dftmp['Motion'] = motionseries.values
                start=0
                end=seq_len

                # Make a temporary .txt file in csv form so we can
                # look at columns
                dftmp.to_csv('tmp.txt', index=False, header=False)
                dfnum = pd.read_csv('tmp.txt', header=None, names=names)
                dftmp=dftmp[:len(dftmp)//seq_len*seq_len]
                # If the person forgot to set the number, then 
                # reset the number for them automatically.
                if(initials_to_number[file_name[0:2]] != dftmp['People'][14]):
                    for i in range(0,len(dftmp)):
                        dftmp.loc[i, 'People'] = initials_to_number[file_name[0:2]]

                df = df.append(dftmp)
        
        df_still=df[df['Motion'] == 0]
        df_still=df_still.reset_index()
        df_still['Index']=df_still['index']
        del df_still['index']
        df_moving=df[df['Motion'] == 1]
        df_moving=df_moving.reset_index()
        df_moving['Index']=df_moving['index']
        del df_moving['index']
        df_still=df_still.reset_index(drop=True)
        df_moving=df_moving.reset_index(drop=True)
        Mean_still=[]
        Std_still=[]
        Mean_moving=[]
        Std_moving=[]

        for i in range(1, 11):
            Mean_still.append(df_still.iloc[:, i].mean())
            Std_still.append(df_still.iloc[:, i].std())
        MeanStd_still=[]
        MeanStd_still.append(Mean_still)
        MeanStd_still.append(Std_still)
        filename = open('meanstd_still.txt', 'w')
        for item in MeanStd_still:
            filename.write("%s\n" % item)
        df_still=standardization(df_still, Mean_still, Std_still)

        for i in range(1, 11):
            Mean_moving.append(df_moving.iloc[:, i].mean())
            Std_moving.append(df_moving.iloc[:, i].std())
        MeanStd_moving=[]
        MeanStd_moving.append(Mean_moving)
        MeanStd_moving.append(Std_moving)
        filename = open('meanstd_moving.txt', 'w')
        for item in MeanStd_moving:
            filename.write("%s\n" % item)
        df_moving=standardization(df_moving, Mean_moving, Std_moving)
        df_still=standardization(df_still, Mean_still, Std_still)

        df=df_still.append(df_moving)
        df=df.sort_values(by='Index')
        df=df.reset_index(drop=True)
        del df['Index']

        df.to_csv('combineddata_train.txt', index=False, header=False)

    if((mode == 'test')or(mode == 'both')or(mode == 'update_phone_graph')):
        if(mode == 'update_phone_graph'):
            pathtest = "./server_local_test_data/"
        else:
            pathtest = "./testdata/"
        all_filestest = glob.glob(os.path.join(pathtest, "*.txt")) #make list of paths
        dftest = pd.DataFrame()

        for file in all_filestest:
            # Getting the file name without extension
            file_name = os.path.splitext(os.path.basename(file))[0]
            if(file_name[0:2] != '00'):        
                # Reading the file content to create a DataFrame
                dftmp = pd.read_csv(file, header=None, names=names, sep=',').convert_objects(convert_numeric=True)
                #drop the first and last line of code which are corrupted by the start/stop action
                dftmp.drop(dftmp.index[:1],inplace = True)
                dftmp.drop(dftmp.tail(1).index,inplace=True)

                dftmp = lowpassfilter(dftmp)
                dftmp=dftmp.dropna(how='any') 
                dftmp.to_csv('tmpdata.txt', index=False, header=False)
                dftmp = pd.read_csv('tmpdata.txt', header=None, names=names)
                ds=difference(dftmp)

                #peak detection using moving avg
                motionth = 0.5 #threshold for identifying motion, difference in number of peaks
                motionlist = motiondetect(ds, motionth)
                motionseries = pd.Series(motionlist)
                dftmp['Motion'] = motionseries.values
                start=0
                end=seq_len

                # Make a temporary .txt file in csv form so we can
                # look at columns
                dftmp.to_csv('tmp.txt', index=False, header=False)
                dfnum = pd.read_csv('tmp.txt', header=None, names=names)
                dftmp=dftmp[:len(dftmp)//seq_len*seq_len]
                # If the person forgot to set the number, then 
                # reset the number for them automatically.
                if(initials_to_number[file_name[0:2]] != dftmp['People'][14]):
                    for i in range(0,len(dftmp)):
                        dftmp.loc[i, 'People'] = initials_to_number[file_name[0:2]]
                dftest = dftest.append(dftmp)
        if(mode != 'both'):
            dftest_still=dftest[dftest['Motion'] == 0]
            dftest_still=dftest_still.reset_index()
            dftest_still['Index']=dftest_still['index']
            del dftest_still['index']
            dftest_moving=dftest[dftest['Motion'] == 1]
            dftest_moving=dftest_moving.reset_index()
            dftest_moving['Index']=dftest_moving['index']
            del dftest_moving['index']
            dftest_still=dftest_still.reset_index(drop=True)
            dftest_moving=dftest_moving.reset_index(drop=True)

            #read the Mean and std that is saved previously from training data
            filename_still = open("meanstd_still.txt")
            MeanStdlist_still = filename_still.readlines()
            Mean_still=MeanStdlist_still[0]
            Mean_still = Mean_still.replace("[", "")
            Mean_still = Mean_still.replace("]", "")
            Mean_still=[float(s) for s in Mean_still.replace("\n", "").split(',')]
            Std_still=MeanStdlist_still[1]
            Std_still = Std_still.replace("[", "")
            Std_still = Std_still.replace("]", "")
            Std_still=[float(s) for s in Std_still.replace("\n", "").split(',')]

            filename_moving = open("meanstd_moving.txt")
            MeanStdlist_moving = filename_moving.readlines()
            Mean_moving=MeanStdlist_moving[0]
            Mean_moving = Mean_moving.replace("[", "")
            Mean_moving = Mean_moving.replace("]", "")
            Mean_moving=[float(s) for s in Mean_moving.replace("\n", "").split(',')]
            Std_moving=MeanStdlist_moving[1]
            Std_moving = Std_moving.replace("[", "")
            Std_moving = Std_moving.replace("]", "")
            Std_moving=[float(s) for s in Std_moving.replace("\n", "").split(',')]
            
        dftest_moving=standardization(dftest_moving, Mean_moving, Std_moving)
        dftest_still=standardization(dftest_still, Mean_still, Std_still)

        dftest=dftest_still.append(dftest_moving)
        dftest=dftest.sort_values(by='Index')
        dftest=dftest.reset_index(drop=True)
        del dftest['Index']

        if(mode == 'update_phone_graph'):
            dftest.to_csv('./server_local_graph/graph_algo_in.txt', index=False, header=False)
        else:
            dftest.to_csv('combineddata_test.txt', index=False, header=False)

        dfHr = dftest.iloc[:,5:13]
        dfHr.to_csv('Hr.txt', index=False, header=False)



    # Delete temporary .txt files to avoid clutter
    os.remove("tmp.txt")
    os.remove("difference.txt")
    os.remove("tmpdata.txt")


    print("done")

if __name__ == '__main__':
    mode = input('do you want to:\n\t-test, train, both or update_phone_graph\n>')
    if((mode == 'test')or(mode == 'train')\
            or(mode == 'both')or(mode == 'update_phone_graph')):
        main(mode)
    else:
        print("invalid input, exiting")