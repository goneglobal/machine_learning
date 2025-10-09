# pip install numpy pandas matplotlib seaborn scikit-learn

# Testing Datasets available in scikit-learn
# There are many built-in datasets in Python:
#   Iris dataset (classic for clustering and classification)
#   Wine dataset (classification)
#   Digits dataset (images of handwritten digits, good for clustering)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target  # Known labels, only for comparison

# Optional: scale features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply k-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster results to a DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = clusters
df['TrueLabel'] = y
print(df.head())

# Visualize clusters (first two features)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("k-Means Clustering on Iris Dataset")
plt.show()
