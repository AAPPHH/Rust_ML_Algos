import numpy as np

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import my_rust_module
import time
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

n_runs   = 100
test_pct = 0.3

sss = StratifiedShuffleSplit(
    n_splits=n_runs, test_size=test_pct, random_state=None)
splits = list(sss.split(X, y))

X_train_list = []
X_test_list  = []
y_train_list = []
y_test_list  = []

for train_idx, test_idx in splits:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)

acc_rust, acc_skle          = [], []
fit_times_rust, fit_times_skle = [], []

# -------- Rust-Loop --------
print("Rust-SVM")
start_rust = time.perf_counter()
for X_train, X_test, y_train, y_test in zip(
        X_train_list, X_test_list, y_train_list, y_test_list):

    rust = my_rust_module.SVM.rbf(gamma=1, c=1.0)

    t0 = time.perf_counter()
    rust.fit(X_train.tolist(), y_train.tolist(),
             max_iter=1_000, tol=1e-3)
    fit_times_rust.append(time.perf_counter() - t0)

    y_pred_rust = np.asarray(rust.predict(X_test.tolist()), dtype=int)
    acc_rust.append(accuracy_score(y_test, y_pred_rust))
end_rust = time.perf_counter()
print(f"Zeit Rust-SVM:   {end_rust - start_rust:.3f} Sekunden")
print(f"Rust-SVM   Acc: {np.mean(acc_rust):.3f} ± {np.std(acc_rust):.3f}")
print(f"Rust-Fit-Zeit   Mittel: {np.mean(fit_times_rust):.4f}s  "
      f"(Σ {np.sum(fit_times_rust):.2f}s)")
print()
# -------- scikit-learn-Loop --------
print("scikit-learn-SVM")
start_skle = time.perf_counter()
for X_train, X_test, y_train, y_test in zip(
        X_train_list, X_test_list, y_train_list, y_test_list):

    sk = SVC(kernel='rbf', gamma=1, C=1.0)

    t0 = time.perf_counter()
    sk.fit(X_train, y_train)
    fit_times_skle.append(time.perf_counter() - t0)

    acc_skle.append(sk.score(X_test, y_test))
end_skle = time.perf_counter()
print(f"Zeit sklearn:    {end_skle  - start_skle:.3f} Sekunden")
print(f"sklearn    Acc: {np.mean(acc_skle):.3f} ± {np.std(acc_skle):.3f}")
print(f"sklearn-Fit-Zeit Mittel: {np.mean(fit_times_skle):.4f}s  "
      f"(Σ {np.sum(fit_times_skle):.2f}s)")
print()
