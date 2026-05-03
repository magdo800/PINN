This Folder is on SAPINNs application to a 1D Poisson Equation, section 4.2 in the dissertation. File contains code SAPINNS_main where the main neural network training experiment is performed. It allows for training with SAPINNs vs standard PINNs. It also contains a sweeps+animate folder which contains: SAPINNS_G_ascent.py that performs a sweep on different learning rates for gradient ascent. SAPINNs_average_over_N_runs.py that compares standard vs SAPINNs for the 50 collocation point problem. SAPINNS_animate_weights.py that creates an animation of the evolution of the SAPINNs during training. Some examples of these animations are contained in Weight_evolution_gifs folder.

SAPINNS_main and SAPINNS_animate_weights.py runs quickly including on CPU (within minutes or less depending on number of epochs). The experiments take longer, depending on how many runs the sweeps are over. 

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
