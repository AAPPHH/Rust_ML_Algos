# Rust SVM – Multiclass Support Vector Machine with PyO3

🚀 **Lightning-fast, standalone SVM implementation in Rust – with seamless Python bindings via [PyO3](https://github.com/PyO3/pyo3).**

---

## 📦 Features

* **Multiclass Support:** One-vs-One (OvO) strategy, identical to scikit-learn
* **Kernel Functions:** Polynomial, RBF, Linear (fully compatible with sklearn parameters)
* **Performance:** Parallelized fitting and prediction powered by Rayon
* **API Compatibility:** `fit`, `predict` methods mirror scikit-learn's SVC
* **Dataset Compatibility:** Easily handles datasets like Iris, Wine, Digits, and more
* **Pure Rust Implementation:** 100% Rust with no external C/FFI dependencies

---

## 🛠️ Technology Stack

* **Rust:** Safe, performant, and reliable
* **Rayon:** Efficient parallel computations
* **PyO3:** Easy-to-use Python interoperability

---

## 🏃 Getting Started

### Prerequisites

* **Rust:** Stable Toolchain ([Install Guide](https://www.rust-lang.org/tools/install))
* **Python:** Version >=3.7 with [pip](https://pip.pypa.io/en/stable/)
* **Maturin:**

  ```bash
  pip install maturin
  ```

### Installation

```bash
git clone <your_repo_url>
cd rust_svm
maturin develop
```

### Python Usage Example

```python
import my_rust_module
# Load your dataset
X_train, y_train = ...
X_test = ...

# Train your model
svm_model = rust_svm.SVM(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)

# Predict outcomes
predictions = svm_model.predict(X_test)
```

---

## 🚧 Roadmap

* [ ] Expand kernel functions
* [ ] GPU acceleration
* [ ] Enhanced compatibility with scikit-learn's advanced features
* [ ] More extensive documentation and examples

---

## 🤝 Contributing

Contributions are warmly welcome! Feel free to open an issue or create a pull request to help enhance compatibility and features.

---

## 📜 License

This project is licensed under the MIT License.

---

Enjoy your Rust-powered SVM journey! 🦀✨
