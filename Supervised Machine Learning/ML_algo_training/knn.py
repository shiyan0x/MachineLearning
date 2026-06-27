from sklearn.neighbors import KNeighborsClassifier

X = [[140, 6],
[180, 7],
[200, 7.5],
[250, 8],
[300, 8.5],
[350, 9],
[360, 9.5]]

y = [0,0,0,1,1,1,1] # 0 = apple  1 = orange

model = KNeighborsClassifier()
model.fit(X,y)

weight = float(input("Enter the weight of the fruit in gram : "))
size = float(input("Enter the size of fruit in the cm: "))

predict = model.predict([[weight, size]])[0]

if predict == 0:
    print("Apple")
else :
    print("orange")
