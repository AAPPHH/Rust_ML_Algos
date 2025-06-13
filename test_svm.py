import my_rust_module
from sklearn.datasets import load_wine
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. Daten laden und skalieren
data = load_wine()
X, y = data.data, data.target
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 2. Modell auf *vollen* skalierten Daten trainieren (ohne PCA)
svm = my_rust_module.SVM.poly(degree=4, gamma=1, coef0=2, c=10.0)
svm.fit(X_scaled.tolist(), y.tolist(), max_iter=500_000, tol=1e-3)

# 3. Für Plot: Projektion der Daten auf 2D mittels PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Grid im 2D-PCA-Raum für Decision-Regions
x_min, x_max = X_pca[:, 0].min() - .5, X_pca[:, 0].max() + .5
y_min, y_max = X_pca[:, 1].min() - .5, X_pca[:, 1].max() + .5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid_points_pca = np.c_[xx.ravel(), yy.ravel()]

# 5. WICHTIG: Gridpunkte aus PCA-Raum zurück in Originalraum für die Vorhersage!
grid_points_orig = pca.inverse_transform(grid_points_pca)
grid_points_orig = grid_points_orig.tolist()

Z = svm.predict(grid_points_orig)
Z = np.array(Z).reshape(xx.shape)

# 6. Visualisierung
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50
)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Wine: Original Labels und Decision Regions (Rust-SVM)')
plt.legend(*scatter.legend_elements(), title="Klasse", loc="upper right")
plt.show()
