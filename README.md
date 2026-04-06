# Nature-Inspired Computation: LSTM Optimization & XAI Tuning

Applying **6 metaheuristic algorithms** to optimize a deep learning model's hyperparameters, then using **4 nature-inspired algorithms** to tune Explainable AI (LIME) parameters — all on the IMDB sentiment classification task.

---

## Overview

This project has two phases:

| Phase | Goal |
|-------|------|
| **Phase 1** | Feature selection (ACA) + LSTM hyperparameter optimization using 6 metaheuristics |
| **Phase 2** | GWO-PSO hybrid for further tuning + XAI parameter optimization with LIME |

---

## Dataset

**IMDB Movie Review Sentiment Classification**
- 50,000 samples (25,000 train / 25,000 test)
- Binary classification: Positive / Negative sentiment
- Vocabulary size: 10,000 words, max sequence length: 256

---

## Phase 1 — Baseline Model & Hyperparameter Optimization

### Baseline LSTM Architecture

```
Embedding(10000, 100, input_length=256)
    → LSTM(units)
    → Dropout(rate)
    → Dense(1, sigmoid)
```

Optimized hyperparameters:
- `lstm_units` ∈ {32, 64, 128, 256}
- `dropout_rate` ∈ {0.1, 0.2, 0.3, 0.4}
- `learning_rate` ∈ {0.01, 0.001, 0.0001}

### Feature Selection — Ant Colony Algorithm (ACA)

Applied ACA proxy to select an optimal vocabulary size K from the full 10,000-word space, modeling fitness as a Gaussian function centered at the estimated optimal K.

### Metaheuristic Algorithms Compared

| Algorithm | Description |
|-----------|-------------|
| Hill Climbing (HC) | Greedy local search over discrete hyperparameter grid |
| Simulated Annealing (SA) | Probabilistic acceptance of worse solutions (α=0.9 cooling) |
| Tabu Search (TS) | Avoids recently visited solutions via tabu list (tenure=3) |
| Genetic Algorithm (GA) | Evolutionary crossover + mutation via DEAP library |
| Particle Swarm (PSO) | Swarm-based global search (w=0.7, c1=1.5, c2=1.5) |
| Firefly Algorithm | Attraction-based movement by brightness (fitness) |

---

## Phase 2 — GWO-PSO Hybrid & XAI Tuning

### GWO-PSO Hybrid
Grey Wolf Optimizer (GWO) used to tune PSO's own parameters (c1, c2, w), then the tuned PSO re-optimizes LSTM hyperparameters — a meta-optimization approach.

### XAI Parameter Optimization with LIME

LIME's `num_features` and `num_samples` parameters were optimized using 4 metaheuristics (HC, SA, GWO, PSO) against a composite score:

```
score = 0.5 × Faithfulness + 0.3 × Stability + 0.2 × Sparsity
```

| XAI Metric | Definition |
|------------|------------|
| Faithfulness | Prediction change when top features are masked |
| Stability | Intersection of top features across 3 runs |
| Sparsity | 1 / number of features used |

**Output:** Top 15 contributing words per sentiment prediction visualized as a horizontal bar chart.

---

## Tech Stack

```
Python · TensorFlow/Keras · DEAP · NumPy · LIME · SHAP · Matplotlib · KaggleHub
```

---

## Project Structure

```
Nature-Inspired-Project/
├── notebook4d0626af83.ipynb   # Full pipeline: Phase 1 + Phase 2
└── README.md
```

---

## How to Run

### Option 1 — Kaggle (Recommended)
The notebook is designed to run on Kaggle with the `jehad24/ni-data` dataset.

### Option 2 — Google Colab
Comment out the Kaggle data loading cell and uncomment the Colab cell:
```python
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
```

Then install dependencies:
```bash
pip install deap lime shap
```
