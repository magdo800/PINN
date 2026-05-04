This Folder is on the Solution Bundle problem, section 5.1 in the dissertation. File contains code Bundles.py where the neural network training experiment is performed.

Bundles.py runs in under a minute on a GPU.  Consider scaling down network width/depth to run quickly on CPU.

Hyperparameters can be changed in the code, in the section under the "Setup" comment.


**Problem (Parametric ODE)**
- ODE:
  - y'(x) + λ y(x) = 0
- Domain:
  - x ∈ [0, 1]
  - λ ∈ [0.5, 3.0]
- Exact solution:
  - y(x,λ) = exp(-λx)

---

**Neural Network (Solution Bundle)**
- input: (x, λ)
- output: y(x, λ)
- architecture:
  - Linear(2 → 64)
  - Tanh
  - 4 hidden layers total
  - Linear(64 → 1)

---

**Model Hyperparameters**
- hidden_units: 64
- depth: 4
- activation: tanh
- hard_ic: True / False (boundary enforcement option)

---

**Initial Condition Handling**
- Soft IC:
  - loss term: (y(0,λ) - 1)^2
  - weight: 10.0
- Hard IC:
  - transformation: y(x,λ) = 1 + x * g(x,λ)
  - satisfies y(0,λ)=1 exactly

---

**Collocation Points (resampled every epoch)**
- x points: N_x = 20
- λ points: N_λ = 20
- total points: N_x × N_λ = 400
- sampling: uniform random

---

**Training**
- epochs: 5000
- optimizer: Adam
- learning_rate: 0.001
- scheduler: CosineAnnealingLR
- batch: full collocation grid

---

**Loss Function**
- residual:
  - R = y'(x) + λ y(x)
- total loss:
  - soft IC: MSE(R,0) + IC_WEIGHT * MSE(IC)
  - hard IC: MSE(R,0)

---

**Test Setup**
- fixed grid (held-out):
  - x: 50 points
  - λ: 50 points
- metric:
  - PDE residual loss on test grid

---

**Evaluation Metrics**
- mean absolute error (MAE)
- max error over x
- error vs λ
- residual loss (train + test)

---

**Experiments**

**1. Soft IC vs Hard IC**
- compare:
  - training loss
  - test loss
  - solution accuracy
  - error heatmaps
- extrapolation:
  - λ ∈ {3.5, 4.0, 4.5, 5.0}

**2. Sample Efficiency**
- N_x = N_λ ∈ {5, 10, 20, 40}
- measure:
  - MAE vs number of collocation points

---

**Evaluation Grid**
- solution heatmaps:
  - 200 × 200 grid (x, λ)
- slice plots:
  - fixed λ values

---

**Outputs**
- train vs test loss curves
- solution plots (exact vs NN)
- heatmaps (exact, prediction, error)
- max error vs λ
- extrapolation plots
- sample efficiency curve

---

**Reproducibility**
- seed: 42
- device: CPU / CUDA
