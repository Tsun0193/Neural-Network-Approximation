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
├── .github/workflows/ci.yml   # GitHub Actions CI
├── Dockerfile       # Docker support
├── docker-compose.yml  # Simplified container startup
├── setup_env.py     # Python-based environment setup
├── requirements.txt
├── README.md
```

---

## ⚙️ Setup Options

### 🧪 Option 1: Native Python (Cross-platform)

```bash
python setup_env.py
```

Then activate virtual environment and launch:

```bash
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.bat   # Windows
jupyter notebook
```

---

### 🐳 Option 2: Docker (Recommended)

#### Build and run the notebook:

```bash
docker-compose up --build
```

Open your browser at: [http://localhost:8888](http://localhost:8888)

---

## ✅ Run Unit Tests

```bash
pytest tests/
```

---

## 📌 CI/CD

GitHub Actions automatically runs tests on push/pull. See `.github/workflows/ci.yml`.