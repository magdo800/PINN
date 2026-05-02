# Physics-Informed Neural Networks — MMath Dissertation

This repository contains all code accompanying the MMath dissertation *Physics-Informed Neural Networks* by Andrey Marshall (Durham University, 2026), supervised by Dr Kasper Peeters.

The dissertation develops the theory of PINNs from first principles and applies them to a range of differential equations, investigating training methodology, failure modes, and advanced techniques. This repository allows you to reproduce every figure in the dissertation and experiment with the hyperparameters yourself.

---

## What are PINNs?

Physics-Informed Neural Networks (PINNs) use neural networks to solve differential equations. Rather than fitting to data, the network is trained by minimising how much it violates the governing equation at a set of sample points (called collocation points). This repository is a practical companion to that idea — if you are new to PINNs, working through the examples in order is a good way to build intuition alongside reading the dissertation.

---

## Repository Structure

Each folder corresponds to one example or experiment from the dissertation and is fully self-contained. Every folder has its own `README.md` with a description of the problem, the network architecture, and a full list of hyperparameters.

```
.
├── lagaris_ode_system/         # Chapter 2 — Coupled ODE system (Lagaris Problem 4)
├── lagaris_poisson_2d/         # Chapter 2 — 2D Poisson equation (Lagaris Problem 6)
├── adaptive_sampling/          # Chapter 3 — RAR and RAD adaptive sampling
├── self_adaptive_pinns/        # Chapter 3 — SA-PINNs with trainable collocation weights
├── spectral_bias_fourier/      # Chapter 3 — Spectral bias and Fourier feature embeddings
├── allen_cahn/                 # Chapter 3 — Allen–Cahn equation with Fourier features
├── solution_bundles/           # Chapter 4 — Parametric ODE families (solution bundles)
├── deeponet_advection/         # Chapter 4 — PI-DeepONet for the 1D advection equation
└── figures/                    # All figures as they appear in the dissertation
```

---

## Dependencies

All examples use **Python 3.10+** and **PyTorch**. The DeepONet example additionally requires **DeepXDE**.

Install the core dependencies with:

```bash
pip install torch torchvision
```

For the DeepONet example only:

```bash
pip install deepxde
```

No other external libraries are required. Standard scientific Python packages (`numpy`, `matplotlib`) are used throughout and will be installed automatically as dependencies of PyTorch.

---

## Hardware

Most examples are lightweight and will run comfortably on a standard laptop CPU in a few minutes. The exception is:

- **`deeponet_advection/`** — this example samples 1000 input functions per epoch and trains for 30,000 iterations. A GPU is strongly recommended. On CPU this will be very slow.

For all other examples, GPU acceleration is optional but will speed things up.

To check whether PyTorch can see your GPU:

```python
import torch
print(torch.cuda.is_available())
```

---

## Getting Started

If you are new to PINNs, the recommended order is:

1. `lagaris_ode_system/` — the simplest example; introduces the core PINN training loop, failure modes, and curriculum learning
2. `lagaris_poisson_2d/` — extends to a 2D PDE; explores collocation strategies and overfitting
3. `adaptive_sampling/` — introduces RAR and RAD for automatic collocation point placement
4. `self_adaptive_pinns/` — trainable weights on collocation points
5. `spectral_bias_fourier/` — demonstrates spectral bias and how Fourier features fix it
6. `allen_cahn/` — a more challenging PDE combining several techniques
7. `solution_bundles/` — learning a parametric family of solutions simultaneously
8. `deeponet_advection/` — operator learning with PI-DeepONets

Each example can also be run independently without working through the others first.

---

## Hyperparameters

Full hyperparameter configurations for every experiment are documented in the `README.md` of each folder. If you want to experiment, the hyperparameters are collected at the top of each script in a clearly labelled configuration block, so they are easy to find and modify.

---

## Citation

If you find this code useful, please cite the dissertation:

```
Andrey Marshall, "Physics-Informed Neural Networks", MMath Dissertation,
Durham University, 2026.
```

---

## Acknowledgements

This work used Durham University's NCC cluster. Thanks to Dr Kasper Peeters for supervision.
