# A temporary file which just take in a file and splits
# it into train and test via a ratio

#to be deleted
import pandas as pd
import numpy as np
names = ['CoughState', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'InstantHr', 'AvgHr','People']
df = pd.read_csv('combineddata.txt', header=None, names=names)



import sys
if(len(sys.argv) != 2):
    print("Error, must supply one arguement, a train\
    test split ratio, a decimal between 0 and 1. \
    \neg: 0.8 would make 80 percent of the data train \
    and 20 percent test.")
#should be between 0 and 1
train_test_ratio = float(sys.argv[1])
if((train_test_ratio > 1) or (train_test_ratio < 0)):
    print("Error: the train test ratio was not between\
    0 and 1. Eg 0.7 = ok, -0.1 = not ok, 1.2 = not ok.")

split_index = int(np.floor(len(df.index)*train_test_ratio))
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]



# also not needed 
print("split index = ", split_index)
print("df.shape = ", df.shape)
print("len(df.index) = ", len(df.index))
print("train_df.shape = ", train_df.shape)
print("test_df.shape = ", test_df.shape)
