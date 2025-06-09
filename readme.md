# Rust SVM – Multiclass Support Vector Machine mit PyO3

**Blitzschnelle, eigenständige SVM-Implementierung in Rust –  
mit Python-Bindings via [PyO3](https://github.com/PyO3/pyo3).**

- **Multiclass:** One-vs-One (OvO), identisch zu scikit-learn
- **Kernels:** Polynomial, RBF, Linear
- **Schnell:** Parallele Fits & Predicts dank Rayon
- **API:** `fit`, `predict` wie bei scikit-learn SVC
- **Kompatibel:** Für Iris, Wine, Digits, uvm.
- **100 % Rust!**

---

## Features

- Poly, RBF und Linear-Kernel (alle Parameter wie bei sklearn)
- Automatische Parallelisierung für viele Klassen und große Daten
- API für scikit-learn-Fans (Python: fast 1:1)
- **Kein C/FFI**: Läuft überall, keine Abhängigkeit von libsvm!
- Leicht zu erweitern um neue Kernel, Optimizer, Features

---

## Installation

### 1. Voraussetzungen

- **Rust**: Stable Toolchain ([Install-Guide](https://www.rust-lang.org/tools/install))
- **Python**: >=3.7, [pip](https://pip.pypa.io/en/stable/)
- **Maturin**:  
  ```bash
  pip install maturin
