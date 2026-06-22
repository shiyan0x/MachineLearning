from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = {
    'Name':['Ali','priya', 'Imran', 'anjali'],
    'Class': ['A', 'B', 'C', 'D'],
    'City': ['Karachi', 'Hyderabad', 'Karachi', 'Hyderabad'],
    'Gender' : ['Male', 'Female', 'Male', 'Female'],
    'Passed' : ['pass', 'Fail', 'Fail', 'pass']
}

df = pd.DataFrame(data)

df_label = df.copy()

le = LabelEncoder()
df_label['Gender_encoder'] = le.fit_transform(df['Gender'])
df_label['Passed_encoder'] = le.fit_transform(df['Passed'])
print("Original Dataframe")
print(df)
print("\n")

print("Label Encoded Dataframe")
print(df_label)

df_encoded = pd.get_dummies(df, columns=['City'], dtype= int)
print("\n")
print("One Hot Encoded Dataframe")
print(df_encoded)
