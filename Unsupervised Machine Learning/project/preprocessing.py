import pandas as pd 
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('StudentPerformance.csv')
print("Missing value in each Column:")
print(df.isnull().sum())

le = LabelEncoder()
df['Internet'] = le.fit_transform(df['Internet'])
df['Passed'] = le.fit_transform(df['Passed'])

print("After Encoding:")
print(df.head())
print('Data type after cleaning')
print(df.dtypes)