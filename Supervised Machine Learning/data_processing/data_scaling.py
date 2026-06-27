import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    'StudyHours' : [1,2,3,4,5],
    'TestScore' : [10,20,30,40,50]
}
df = pd.DataFrame(data)

#standard scaler
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)
print("standerd scaler output")
print(pd.DataFrame(standard_scaled, columns=['StudyHours', 'TestScore']))

#min max scaler
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)
print("\nminmac scaled output")
print(pd.DataFrame(minmax_scaled, columns=['StudyHours', 'TestScore']))

X = df[['StudyHours']]
Y = df[['TestScore']]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

print("training data")
print(X_train)
print("testing data")
print(X_test)

print("training data")
print(Y_train)
print("testing data")
print(Y_test)

