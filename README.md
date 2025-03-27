# Neural-Network-Approximation

This project explores the theoretical and practical limits of approximating univariate convex functions using piecewise-linear neural networks (ReLU-based).

---

## 🧠 Project Goals

- Approximate convex functions using ReLU networks
- Evaluate performance across different architectures
- Compare with theoretical approximation bounds
- Understand the limitations of classical neural regressors

---

## 📁 Structure

```
project/
├── data/           # Data generators
├── model/          # Network definitions (ReLU, piecewise, etc.)
├── utils/          # Training, evaluation, visualizers
├── tests/          # Unit tests for CI/CD
├── experiments/
│   └── report.ipynb   # Main notebook with theory & results
```

---

## ⚙️ Setup

### 🔧 Linux / macOS

```bash
bash setup.sh
```

### 🪟 Windows

```bat
setup.bat
```

---

## 🚀 Quick Start

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
jupyter notebook
```

Then open `experiments/report.ipynb`.

---

## ✅ Run Unit Tests

```bash
pytest tests/
```

---

## 📌 CI/CD

GitHub Actions automatically runs tests on push/pull. See `.github/workflows/ci.yml`.