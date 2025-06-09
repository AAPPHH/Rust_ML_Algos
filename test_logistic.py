import my_rust_module
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

clf = my_rust_module.LogisticRegression()
clf.fit(X_pca.tolist(), y.tolist(), n_iter=300, lr=0.1)

x_min, x_max = X_pca[:, 0].min() - .5, X_pca[:, 0].max() + .5
y_min, y_max = X_pca[:, 1].min() - .5, X_pca[:, 1].max() + .5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = clf.predict(grid_points.tolist())
Z = np.array(Z).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50
)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Iris: Original Labels und Decision Regions (Rust-LogisticRegression)')
plt.legend(*scatter.legend_elements(), title="Klasse", loc="upper right")
plt.show()
