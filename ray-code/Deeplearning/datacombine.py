# loading libraries
import pandas as pd
import glob
import os

names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People']


path = "./data/"
all_files = glob.glob(os.path.join(path, "*.txt")) #make list of paths
df = pd.DataFrame()

initials_to_number = {"aw":0.0, "sc":1.0, "lj":2.0,\
                        "ls":3.0, "ir":4.0, "ik":5.0,\
                        "sa":6.0}
names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People']


for file in all_files:
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    if(file_name[0:2] != '00'):        
        # Reading the file content to create a DataFrame
        dftmp = pd.read_csv(file, header=None, names=names)
        dftmp.drop(dftmp.index[:1],inplace = True)
        dftmp.drop(dftmp.tail(1).index,inplace=True)

        # Make a temporary .txt file in csv form so we can
        # look at columns
        dftmp.to_csv('tmp.txt', index=False, header=False)
        dfnum = pd.read_csv('tmp.txt', header=None, names=names)

        # If the person forgot to set the number, then 
        # reset the number for them automatically.
        if(initials_to_number[file_name[0:2]] != dfnum['People'][14]):
            for i in range(0,len(dfnum)):
                dfnum.loc[i, 'People'] = initials_to_number[file_name[0:2]]

        df = df.append(dfnum)

# Delete temporary .txt file for reseting the person number
os.remove("tmp.txt")

# Export data to combineddata file with correct numbers
df.to_csv('combineddata_test.txt', index=False, header=False)
print("done")