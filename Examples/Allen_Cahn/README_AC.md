

**Problem (Allen–Cahn PDE)**
- PDE:
  - u_t - d*u_xx - 5*(u - u^3) = 0
- Domain:
  - x ∈ [-1, 1]
  - t ∈ [0, 1]
- Parameter:
  - d = 0.0001

**Initial Condition**
- u(x,0) = x² cos(πx)

**Boundary Conditions**
- u(-1,t) = -1
- u(1,t)  = -1

---

**Neural Networks**
- Two models:
  - Vanilla PINN (MLP)
  - Fourier Feature PINN

**Vanilla PINN**
- input: (x, t)
- architecture:
  - Linear(2 → width)
  - Tanh
  - Linear(width → width)
  - Tanh
  - Linear(width → 1)

**Fourier PINN**
- Fourier features:
  - num_fourier: 256
  - scale (sigma): 2.0
- embedding:
  - input → 2M features (sin + cos)
- architecture:
  - Linear(512 → width)
  - Tanh
  - Linear(width → width)
  - Tanh
  - Linear(width → 1)

---

**Model Hyperparameters**
- hidden_layers: 1
- width: 10
- activation: tanh
- initialization: Xavier uniform

---

**Collocation Points (resampled every iteration)**
- domain points: 8000
- IC points: 800
- BC points: 400

---

**Training**
- optimizer: Adam
- learning_rate: 0.001
- scheduler: exponential decay
- decay_rate: 1
- decay_steps: 5000
- iterations: 50000

---

**Loss Function**
- PDE residual:
  - R = u_t - d*u_xx - 5*(u - u^3)
- total loss:
  - loss = MSE(R, 0) + IC_loss + BC_loss

---

**Hard Constraint Option**
- USE_HARD: False (default)
- if True:
  - IC + BC enforced analytically via transform
  - no IC/BC loss terms

---

**Fourier Feature Parameters**
- num_fourier: 256
- output_dim after embedding: 512
- sigma: 2.0

---

**Evaluation**
- reference dataset: Allen_Cahn.mat
- metric:
  - relative L2 error

---

**Logging**
- log_every: 500 steps
- tracked:
  - total loss
  - PDE loss
  - IC loss
  - BC loss
  - L2 error

---

**Outputs**
- loss vs iteration
- L2 error vs iteration
- solution heatmaps (true vs predicted)
- absolute error heatmap
- solution slices at fixed times
- vanilla vs Fourier comparison

---

**Reproducibility**
- seed: 22
- device: CPU / CUDA
