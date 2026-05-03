This Folder is on SAPINNs application to a 1D Poisson Equation, section 4.2 in the dissertation. File contains code SAPINNS_main where the main neural network training experiment is performed. It allows for training with SAPINNs vs standard PINNs. It also contains a sweeps+animate folder which contains: resample_frequency.py that tests convergence over different resampling frequencies, and sampling_method that compares over the different sampling types for both fixed and resampling (as outlined in the dissertation).

LagQ6_main runs quickly including on CPU (within minutes or less depending on number of epochs). The experiments take about 30x longer.



**Hyperparameters:**

**Problem**
- 1D Poisson-type PDE:
  - -u_xx = f(x)
- Domain:
  - x ∈ (0, 1)
- Parameters:
  - eps = 2.5
  - beta = 10
  - x0 = 0.3

**Exact Solution**
- u(x) = sin(6πx) + eps * exp(-beta(x - x0)^2)

**Forcing Function**
- f(x) derived analytically from u(x)

---

**Neural Network (PINN / SA-PINN)**
- input: x
- output: u(x)
- architecture:
  - Linear(1 → 64)
  - Tanh
  - Linear(64 → 64)
  - Tanh
  - Linear(64 → 1)

---

**Collocation Points**
- N_r = 50
- sampling: uniform random in (0,1)
- used for physics residual loss

**Boundary Points**
- x_b = [0, 1]
- Dirichlet boundary condition enforced via MSE loss

**Test Grid**
- 500 uniformly spaced points in (0,1)

---

**Training Setup**
- epochs: 20000
- optimizer: Adam
- learning_rate: 0.001
- loss: MSE

---

**Vanilla PINN Loss**
- L = mean(R^2) + mean(boundary_error^2)

---

**SA-PINN (Self-Adaptive PINN)**
- same network as PINN
- additional learnable weights:
  - λ_r (collocation weights)
  - λ_b (boundary weights)

**SA update rule**
- λ_r ← λ_r + lr_λ * R^2
- λ_b ← λ_b + lr_λ * B^2
- normalization: divide by mean

---

**SA Hyperparameters**
- lambda_r init: 1 (size: 50×1)
- lambda_b init: 1 (size: 2×1)
- lr_lambda: 0.01

---

**Evaluation Metrics**
- relative L2 error:
  - ||u_pred - u_true|| / ||u_true||
- tracked every epoch

---

**Outputs / Plots**
- relative error vs epoch (PINN vs SA-PINN)
- solution comparison (true vs predictions)
- pointwise absolute error
- residual plot (SA-PINN)
- SA weight distribution (λ_r)

---

**Reproducibility**
- seed: 0
- device: CPU / CUDA (auto-selected)
