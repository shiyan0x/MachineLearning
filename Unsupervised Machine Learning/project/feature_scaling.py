import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('StudentPerformance.csv')

le = LabelEncoder()
df['Internet'] = le.fit_transform(df['Internet'])
df['Passed'] = le.fit_transform(df['Passed'])

features = ['StudyHours', 'Attendance', 'SleepHours', 'PastScore']
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

X = df_scaled[features] #features
y = df_scaled['Passed'] #target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Classification Report:\n', classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', 
            xticklabels = ['Fail', 'Passed'], yticklabels = ['Fail', 'Passed'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual')
plt.title('Confusion Matrix ')
plt.tight_layout()
plt.show()

print("----Predicted Results----\n")
try:
    study_hours = float(input("Enter Study hours: "))
    sleep_hours = float(input("Enter Sleep hours: "))
    attendance = float(input("Enter Attendance: "))
    past_score = float(input("Enter Past score: "))

    user_input_df = pd.DataFrame([{
        'StudyHours' : study_hours,
        'Attendance' : attendance,
        'SleepHours' : sleep_hours,
        'PastScore' : past_score
    }])

    user_input_scaled = scaler.transform(user_input_df)
    # Convert numpy array back to DataFrame with feature names
    user_input_scaled_df = pd.DataFrame(user_input_scaled, columns=features)
    prediction = model.predict(user_input_scaled_df)[0]

    
    result = "PASS" if prediction == 1 else "FAIL"
    print(f"Prediction Based on input: {result}")

except Exception as e:
    print("An error occured: ",e)

    
    
