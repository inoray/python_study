import csv as csv
import numpy as np
import pandas as pd

csv_file_object = csv.reader(open('train.csv'))
next(csv_file_object)
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

#print(data)
#print(data[0:15, 5])

#type(data[::, 5])
#np.ndarray

#ages_onboard = data[0::, 5].astype(np.float)

df = pd.read_csv('train.csv', header=0)

#print(df)

print(df["Survived"].value_counts(normalize=True))

dfSex = df["Sex"] == 'male'
print(dfSex)

dfSS = df['Survived'][df['Sex'] == 'male'].value_counts(normalize=True)
print(dfSS)