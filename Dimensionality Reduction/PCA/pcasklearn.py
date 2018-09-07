import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


pc_data = pd.read_csv('/Users/manojrajalbandi/Desktop/Study/ML/hw3/pca-data.txt', sep = '\t', header = None)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pc_data.values)


print("Original 3-Dimension Data\n")
print(pc_data.values)

print("Original Scaled Data\n")
print(scaled_data)

sklearn_pca = PCA(n_components=2)
reduced_data = sklearn_pca.fit_transform(pc_data.values)


print("Reduced 2-Dimension Data\n")
print (reduced_data)

print("Original Data Dimension\n")
print(pc_data.shape)


print("Reduced Data Dimension\n")
print(reduced_data.shape)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label='Original points reduced')
plt.title("PCA sklearn")
plt.legend()
plt.show()
