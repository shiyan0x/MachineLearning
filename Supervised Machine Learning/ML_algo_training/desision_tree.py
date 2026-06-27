from sklearn.tree import DecisionTreeClassifier

X = [[7,2],
[8,3],
[9,8],
[10,9]]

y = [0,0,1,1] # 0 = apple  1 = orange

model = DecisionTreeClassifier()
model.fit(X,y)

size = float(input("Enter the size of fruit in the cm: "))
color = float(input("Enter the color (1-10) : "))

predict = model.predict([[size, color]])[0]

if predict == 0:
    print("Apple")
else :
    print("orange")
