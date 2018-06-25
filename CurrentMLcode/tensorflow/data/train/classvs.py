import pandas as pd

names=['Class']
df = pd.read_csv('y_train.txt', header=None, names=names)
print(df['Class'].value_counts())