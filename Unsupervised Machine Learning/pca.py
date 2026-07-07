from sklearn.metrics import explained_variance_score
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    'Age' : [25, 30, 35,40 ,45],
    'Income' : [30000, 40000, 50000, 60000, 70000],
    'Spending' : [29000, 35000, 42000, 50000, 55000],
    'Saving' : [1000, 5000, 8000, 10000, 15000]
}

df = pd.DataFrame(data)
scaler = StandardScaler()
Scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(Scaled_data)

pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])

explained_variance = pca.explained_variance_ratio_
print("Variance captured by each PCA components")
print(np.round(explained_variance * 100, 2))

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], color='black', s=80)
plt.title("PCA Projection (2D view)")
plt.xlabel('PCA1 Main Pattern')
plt.ylabel('PCA2 Minor Pattern')
plt.grid(True)
plt.show()

