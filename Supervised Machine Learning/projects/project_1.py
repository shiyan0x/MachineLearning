from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd 
 
# 1. load data    
data = pd.read_csv("StudentPerformanceFactors.csv")

# 2. select features and target   
X = data[['Hours_Studied']]
y = data['Exam_Score']

# 3. train the model
model = LinearRegression()
model.fit(X, y)
model_prediction = model.predict(X)

# 4. evaluate the model
mae = mean_absolute_error(y, model_prediction)
mse = mean_squared_error(y, model_prediction)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# 5. take input and predict
new_hours = float(input("enter the hours_studied: "))
predicted_score = model.predict(pd.DataFrame({"Hours_Studied":[new_hours]}))
print(f"The predicted score for {new_hours} hours studied is {predicted_score[0]}: ")