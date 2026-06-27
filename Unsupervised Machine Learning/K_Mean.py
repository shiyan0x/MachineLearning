import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Sample data
data = {
    'Customer' : ['Riya', 'Aman','Faizan','Aarav','Ishan', 'kesu'],
    'Age' : [20,21,45,20,38,40],
    'Spending' : [80,70,200,85,250,300]
}
df = pd.DataFrame(data)
X = df[['Age', 'Spending']]
model = KMeans(n_clusters = 2, random_state=42, n_init = 10)
df['Group']= model.fit_predict(X)
plt.figure(figsize= (6,5))
for group in df['Group'].unique():
    group_data = df[df['Group']==group]
    plt.scatter(group_data['Age'], group_data['Spending'], label=f'Group{group}')

plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Customer Segments (k-Means)')
plt.legend()
plt.grid(True)
plt.show()
print(df)