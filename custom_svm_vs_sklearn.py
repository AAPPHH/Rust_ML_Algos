import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import my_rust_module

iris = load_iris()
X, y = iris.data, iris.target

n_runs   = 100
test_pct = 0.3
pca      = PCA(n_components=2, random_state=42)

acc_rust = []
acc_skle = []

sss = StratifiedShuffleSplit(
    n_splits=n_runs, test_size=test_pct, random_state=123
)

for split_idx, (train_idx, test_idx) in enumerate(sss.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    rust = my_rust_module.SVM.poly(degree=4, gamma=1, coef0=2.0, c=1.0)
    rust.fit(X_train_pca.tolist(), y_train.tolist(),
             max_iter=30000, tol=1e-3)
    y_pred_rust = np.asarray(rust.predict(X_test_pca.tolist()), dtype=int)
    acc_rust.append(accuracy_score(y_test, y_pred_rust))

    sk  = SVC(kernel='poly', degree=4, gamma=1, coef0=2.0, C=1.0)
    sk.fit(X_train_pca, y_train)
    acc_skle.append(sk.score(X_test_pca, y_test))

print(f"Rust-SVM   Mittel ± Std: {np.mean(acc_rust):.3f} ± {np.std(acc_rust):.3f}")
print(f"sklearn    Mittel ± Std: {np.mean(acc_skle):.3f} ± {np.std(acc_skle):.3f}")
