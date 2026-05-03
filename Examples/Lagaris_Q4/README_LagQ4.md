**Problem**
- ODE system:
  - psi1'(x) = cos(x) + psi1^2 + psi2 - (1 + x^2 + sin^2(x))
  - psi2'(x) = 2x - (1 + x^2)sin(x) + psi1*psi2
- Exact solution:
  - psi1(x) = sin(x)
  - psi2(x) = 1 + x^2
- Domain:
  - x ∈ (0, 3)

**Boundary Conditions**
- psi1(0) = 0
- psi2(0) = 1

**Neural Network**
- input: x
- output: N1(x), N2(x)
- hidden layers: 1
- hidden units: 10
- activation: tanh
- init: Xavier uniform

**Trial Solution**
- psi1(x) = x * N1(x)
- psi2(x) = 1 + x * N2(x)

**Collocation Points**
- num_samples: 10
- x_range: (0, 3)
- sampling: uniform grid

**Architecture**
- Linear(1 → 10) → tanh → Linear(10 → 2)

**Training**
- optimizer: Adam
- learning_rate: 0.001
- loss: MSELoss
- epochs: 15000
- batch_size: 10

**Loss (Physics Residual)**
- eq1 residual: dpsi1/dx - RHS1
- eq2 residual: dpsi2/dx - RHS2
- total: MSE(res1, 0) + MSE(res2, 0)

**Evaluation**
- solution points: 30
- residual grid: 2000 points

**Outputs**
- psi1_NN, psi2_NN
- absolute error vs exact
- residuals vs x
- training + test loss

**Reproducibility**
- seed: 2
- deterministic: True
- cudnn deterministic: True
