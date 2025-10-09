# pip install numpy pandas matplotlib seaborn scikit-learn mlxtend

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

# Association rules
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# 1. Clustering Example (Iris dataset)
# -----------------------------
print("=== Clustering Example ===")
iris = load_iris()
X = iris.data
y_true = iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Silhouette score
score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {score:.2f}")

# Plot clusters (first two features)
plt.figure(figsize=(6,4))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='viridis', s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("k-Means Clustering on Iris Dataset")
plt.show()

# -----------------------------
# 2. Association Rules Example (Market Basket)
# -----------------------------
print("\n=== Association Rules Example ===")
# Small example dataset
transactions = [
    ['bread','milk'],
    ['bread','diaper','beer','egg'],
    ['milk','diaper','beer','cola'],
    ['bread','milk','diaper','beer'],
    ['bread','milk','diaper','cola'],
]

# Convert to one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# Frequent itemsets
frequent_itemsets = apriori(df_trans, min_support=0.6, use_colnames=True)
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[['antecedents','consequents','support','confidence','lift']])

# -----------------------------
# 3. Feature Selection Example (Wine dataset)
# -----------------------------
print("\n=== Feature Selection Example ===")
wine = load_wine()
X_wine = wine.data
y_wine = wine.target
feature_names = wine.feature_names

# Select top 5 features using ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X_wine, y_wine)
selected_features = np.array(feature_names)[selector.get_support()]
print("Top 5 selected features:", selected_features)

# -----------------------------
# 4. Anomaly Detection Example (Isolation Forest)
# -----------------------------
print("\n=== Anomaly Detection Example ===")
# Use Iris dataset again
iso = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso.fit_predict(X_scaled)

# -1 = anomaly, 1 = normal
df_iris = pd.DataFrame(X_scaled, columns=iris.feature_names)
df_iris['Anomaly'] = y_pred
print(df_iris['Anomaly'].value_counts())

# Plot anomalies
plt.figure(figsize=(6,4))
sns.scatterplot(x=df_iris[iris.feature_names[0]], 
                y=df_iris[iris.feature_names[1]], 
                hue=df_iris['Anomaly'], palette=['red','green'])
plt.title("Anomaly Detection with Isolation Forest")
plt.show()
