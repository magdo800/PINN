This Folder is on the PI-DeepONet, finding the operator mapping the initial condition to the solution for the advection equation, "PI-DeepONet", section 5.2 in the dissertation. File contains code DeepO.py where the neural network training experiment is performed.

DeepO.py takes 1-2 hours on a GPU. Consider scaling down

Hyperparameters can be changed in the code, in the section under the "Setup" comment. Consider scaling down problem (network size, number of points and epochs) to run in a reasonable amount of time on a CPU.

---

## Problem

We learn the mapping  
v(x) → u(x,t)

where:

- PDE:  
  u_t + u_x = 0,  (x,t) ∈ [0,1]^2

- Initial condition:  
  u(x,0) = v(x)

- Exact solution:  
  u(x,t) = v(x - t)

The input v(x) is sampled from a **periodic Gaussian random field (GRF)**.

---

## What this script does

- Trains a **PI-DeepONet** to learn the operator v → u  
- Evaluates generalisation over unseen functions  
- Produces all figures used in Section 5.2  

---

## Model

DeepONet architecture:

- **Branch net** → values of v(x) at sensor points  
- **Trunk net** → coordinates (x, t)  
- Output → u(x,t)

Physics enforced via:
u_t + u_x = 0

---

## Hyperparameters

### Function space
- GRF kernel → ExpSineSquared (periodic)
- length_scale → controls smoothness

### Sensors
- num_sensors → number of evaluation points for v(x)

### Collocation points
- num_domain → PDE collocation points  
- num_boundary → initial condition points  

### Network
- branch → [num_sensors, 128, 128, 128]  
- trunk → [2, 128, 128, 128]  
- activation → tanh  
- init → Glorot normal  

### Training
- optimiser → Adam  
- learning rate → 5e-4  
- iterations → e.g. 30000  
- batch size → 32  

---

## Outputs

Saved to ./figures/:

- fig1 → GRF samples + exact solutions  
- fig2 → training loss (train vs test)  
- fig3 → prediction vs ground truth + error  
- fig4 → L2 error distribution  
- fig5 → error vs number of sensors  

---

## Experiments

### Main run
- 50 sensors  
- 30k iterations  

### Sensor ablation
- sensors ∈ [10, 20, 30, 50, 75, 100]  
- retrain and evaluate L2 error  

---

## Notes

- Collocation points are sampled internally by DeepXDE  
- Periodicity handled via:
  - GRF kernel  
  - trunk feature transform (sin/cos)  
- Learns an **operator**, not a single solution  

---

## Minimal mental model

- PINN → learns one solution  
- DeepONet → learns function → function  
- PI-DeepONet → adds PDE constraint  
