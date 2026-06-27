from sklearn.linear_model import LinearRegression

X = [[1],[2],[3],[4],[5]]
y = [0,0,0,1,1]
model = LinearRegression()

model.fit(X,y)

hours = float(input("Enter the learning hours: "))
result = model.predict([[hours]])[0]

if result == 1:
    print("pass")
else :
    print("fail")