# loading libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
import operator
import math
import glob
import os

names = ['Cough state', 'EMG1', 'EMG2', 'Vibration1', 'Vibration2', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Hr', 'Instant Hr', 'Avg Hr','People']


path = "./data/"
all_files = glob.glob(os.path.join(path, "*.txt")) #make list of paths
df = pd.DataFrame()

for file in all_files:
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    # Reading the file content to create a DataFrame
    dftmp = pd.read_csv(file, header=None, names=names)
    dftmp.drop(dftmp.index[:1],inplace = True)
    dftmp.drop(dftmp.tail(1).index,inplace=True)
    df = df.append(dftmp)

df.to_csv('combineddata.txt', index=False, header=False)
print("done")