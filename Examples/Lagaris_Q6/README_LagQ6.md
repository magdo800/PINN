This Folder is on Lagaris Question 6 "A Two Dimensional Poisson Equation", section 3.2, in the dissertation. File contains code LagQ6_main where the main neural network training experiment is performed. It also contains a Sampling_experiments folder which contains: resample_frequency.py that tests convergence over different resampling frequencies, and sampling_method that compares over the different sampling types for both fixed and resampling (as outlined in the dissertation).

LagQ6_main runs quickly including on CPU (within minutes or less depending on number of epochs). The experiments take about 30x longer. 

Hyperparameters in LagQ6_main can be changed in the section under the "Run" comment.

**Problem (PDE)**
- 2D Poisson-type PDE:
  - ∇²ψ(x,y) = f(x,y)
- Domain:
  - x ∈ (0, 1)
  - y ∈ (0, 1)
- Parameter:
  - a = 3

**Exact Solution**
- ψ(x,y) = exp(-(a*x + y)/5) * sin(a²x² + y)

**Neural Network**
- input: (x, y)
- output: N(x,y)
- hidden layers: 1
- hidden units: 10
- activation: tanh
- initialization: Xavier uniform

**Trial Solution (BC enforced)**
- ψ_trial(x,y) = A(x,y) + x(1-x)y(1-y) * N(x,y)

**Boundary Handling**
- A(x,y): constructed to satisfy Dirichlet boundary conditions exactly
- ensures ψ matches exact solution on all edges

**Collocation Points**
- num_samples: 10
- total points: num_samples² (grid)
- domain: (0,1) × (0,1)
- sampling: uniform grid or optional random sampling per epoch

**Architecture**
- Linear(2 → 10)
- activation: tanh
- Linear(10 → 1)

**Training**
- optimizer: Adam
- learning_rate: 0.001
- loss: MSELoss (physics residual)
- epochs: 10000 current setting in code (50000 in dissertation)
- batch_size: full dataset

**Physics Loss**
- residual = ∇²ψ_trial - f(x,y)
- loss = MSE(residual, 0)

**Evaluation Grid**
- test resolution: 30 × 30
- training grid: 10 × 10
- residual grid: same as test grid

**Outputs**
- ψ_NN(x,y)
- ψ_exact(x,y)
- absolute error |NN - exact|
- PDE residual field
- train loss per epoch
- test loss per checkpoint

**Metrics**
- mean absolute error (train grid)
- mean absolute error (test grid)
- physics residual loss

**Training Modes**
- deterministic collocation grid (default)
- optional random sampling per epoch

**Reproducibility**
- seed: 2
- deterministic execution enabled
