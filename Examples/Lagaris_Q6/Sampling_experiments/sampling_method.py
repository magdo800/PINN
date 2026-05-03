# -*- coding: utf-8 -*-
"""
LagQ6: Sampling Methods experiment for Lagaris Problem 6, section 3.2 in dissertation.
Sampling Method Experiment
==========================

This experiment compares different collocation sampling strategies in a
Physics-Informed Neural Network (PINN) setting.

Two regimes are studied:

A) Fixed sampling
   - Collocation points are drawn once and kept constant

B) Resampling
   - Collocation points are periodically regenerated every N epochs

The goal is to analyze how sampling strategy affects:
    - convergence speed
    - final error
    - stability across random seeds

All results are saved for reproducibility and post-analysis.
"""

import torch
import torch.utils.data
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file saving
import matplotlib.pyplot as plt
import random
import time
import os
from scipy.stats import qmc


# ================================================================== #
#  Sampling helpers
# ================================================================== #

def generate_collocation_pts(method: str, n: int, seed: int = 0,
                             skip: int = 0) -> torch.Tensor:
    """
    Generate collocation points in the unit square [0,1]^2.

    Each method produces a different spatial distribution:
        - grid: structured deterministic mesh
        - random: uniform i.i.d. sampling
        - lhs: Latin Hypercube Sampling
        - halton: low-discrepancy sequence
        - hammersley: quasi-random sequence
        - sobol: base-2 quasi-random sequence

    Parameters
    ----------
    method : sampling strategy
    n      : number of points
    seed   : randomness seed (for stochastic methods)
    skip   : offset for quasi-random sequences (ensures non-overlap)
    """

    # ---------------------------- GRID ---------------------------- #
    if method == 'grid':
        # Build equispaced grid (sqrt(n) × sqrt(n))
        side = int(np.ceil(np.sqrt(n)))
        xs = torch.linspace(0., 1., side)
        ys = torch.linspace(0., 1., side)
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')
        pts = torch.stack([gx.flatten(), gy.flatten()], dim=1).float()
        return pts[:n]  # trim excess

    # --------------------------- RANDOM --------------------------- #
    elif method == 'random':
        # Fully stochastic uniform sampling
        rng = torch.Generator()
        rng.manual_seed(seed)
        return torch.rand(n, 2, generator=rng)

    # ---------------------------- LHS ---------------------------- #
    elif method == 'lhs':
        # Latin Hypercube ensures stratified coverage per dimension
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        samples = sampler.random(n)
        return torch.tensor(samples, dtype=torch.float32)

    # --------------------------- HALTON --------------------------- #
    elif method == 'halton':
        # Low-discrepancy sequence with deterministic structure
        sampler = qmc.Halton(d=2, scramble=False)
        sampler.fast_forward(skip)
        samples = sampler.random(n)
        return torch.tensor(samples, dtype=torch.float32)

    # ------------------------- HAMMERSLEY ------------------------ #
    elif method == 'hammersley':
        # Mix of deterministic linear spacing + Halton sequence
        halton_sampler = qmc.Halton(d=1, scramble=False)
        halton_sampler.fast_forward(skip)
        halton = halton_sampler.random(n)

        # First dimension: uniform spacing within current block
        first = np.arange(1, n + 1).reshape(-1, 1) / n

        samples = np.hstack([first, halton])
        return torch.tensor(samples, dtype=torch.float32)

    # ---------------------------- SOBOL --------------------------- #
    elif method == 'sobol':
        # Base-2 low-discrepancy sequence
        m = int(np.ceil(np.log2(max(skip + n, 1))))
        sampler = qmc.Sobol(d=2, scramble=False)
        all_pts = sampler.random_base2(m)
        samples = all_pts[skip:skip + n]

        # Safety fallback if indexing exceeds sequence length
        if len(samples) < n:
            samples = all_pts[:n]

        return torch.tensor(samples, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown sampling method: {method}")


# ================================================================== #
#  Dataset
# ================================================================== #

class DataSet(torch.utils.data.Dataset):
    """
    Simple 2D grid dataset used for baseline training (fixed grid mode).
    """
    def __init__(self, num_samples, x_range, y_range):
        x = torch.linspace(x_range[0], x_range[1], num_samples)
        y = torch.linspace(y_range[0], y_range[1], num_samples)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        self.data_in = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        self.x_range, self.y_range = x_range, y_range
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, i):
        return self.data_in[i]


# ================================================================== #
#  Neural Network
# ================================================================== #

class Fitter(torch.nn.Module):
    """
    Fully connected neural network used as PINN trial function.
    """
    def __init__(self, num_hidden_nodes, num_layers=1):
        super().__init__()
        assert num_layers >= 1

        # Input layer
        layers = [torch.nn.Linear(2, num_hidden_nodes), torch.nn.Tanh()]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers += [torch.nn.Linear(num_hidden_nodes, num_hidden_nodes), torch.nn.Tanh()]

        # Output layer
        layers.append(torch.nn.Linear(num_hidden_nodes, 1))

        self.net = torch.nn.Sequential(*layers)

        # Xavier initialization for stability
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


# ================================================================== #
#  PINN Solver
# ================================================================== #

class PinnSolver:
    """
    Physics-Informed Neural Network training engine.

    Handles:
        - sampling strategy
        - PDE residual computation
        - training loop
        - evaluation metrics
    """
    def __init__(self, network, epochs, loss_fn, optimiser,
                 sampling_method='grid', num_collocation_pts=100,
                 resample_every=None,
                 x_range=(0, 1), y_range=(0, 1),
                 a=3, seed=0, plot_samples=30,
                 train_loader=None):

        # Core components
        self.network             = network
        self.epochs              = epochs
        self.loss_fn             = loss_fn
        self.optimiser           = optimiser

        # Sampling configuration
        self.sampling_method     = sampling_method
        self.num_collocation_pts = num_collocation_pts
        self.resample_every      = resample_every

        # Domain setup
        self.x_range = x_range
        self.y_range = y_range

        # PDE parameter
        self.a = a

        # Reproducibility
        self.seed = seed

        # Evaluation resolution
        self.plot_samples = plot_samples

        # Optional fixed-grid loader
        self.train_loader = train_loader

        # Internal sampling state
        self._cached_pts = None
        self._resample_seed_counter = seed
        self._resample_count = 0

        # Training logs
        self.epoch = 0
        self.loss = 0.
        self.loss_list = []
        self.test_loss_list = []
        self.test_epochs = []

        # True error tracking
        self.train_true_error_list = []
        self.test_true_error_list = []
        self.true_error_epochs = []

        # ---------------- test grid ---------------- #
        n = self.plot_samples
        xs = torch.linspace(*self.x_range, n)
        ys = torch.linspace(*self.y_range, n)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        self._test_xf = X.flatten().unsqueeze(1)
        self._test_yf = Y.flatten().unsqueeze(1)

        # ---------------- train grid (true error) ---------------- #
        xst = torch.linspace(*self.x_range, 10)
        yst = torch.linspace(*self.y_range, 10)
        Xt, Yt = torch.meshgrid(xst, yst, indexing='ij')
        self._train_xf = Xt.flatten().unsqueeze(1)
        self._train_yf = Yt.flatten().unsqueeze(1)

    # ========================================================== #
    #  PDE + exact solution
    # ========================================================== #

    def psi_exact(self, x, y):
        a = self.a
        return torch.exp(-(a * x + y) / 5) * torch.sin(a**2 * x**2 + y)

    def A(self, x, y):
        # boundary correction term ensuring exact BC satisfaction
        A_val = (
            (1 - x) * self.psi_exact(torch.zeros_like(x), y)
            + x * self.psi_exact(torch.ones_like(x), y)
        )
        return A_val

    def psi_trial(self, x, y):
        xy = torch.cat([x, y], dim=1)
        N = self.network(xy)
        return self.A(x, y) + x * (1 - x) * y * (1 - y) * N

    def compute_residual(self, x, y):
        # PDE residual via autograd
        psi = self.psi_trial(x, y)
        dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        dpsi_dy = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True)[0]
        d2psi_dy2 = torch.autograd.grad(dpsi_dy, y, grad_outputs=torch.ones_like(dpsi_dy), create_graph=True)[0]

        laplacian = d2psi_dx2 + d2psi_dy2

        f = torch.sin(x + y)  # simplified placeholder forcing term
        return laplacian - f

    # ========================================================== #
    #  Training
    # ========================================================== #

    def _get_collocation_pts(self):
        # decide whether to resample
        need = (
            self._cached_pts is None
            or (self.resample_every is not None and self.epoch % self.resample_every == 0)
        )

        if need:
            n = self.num_collocation_pts

            if self.sampling_method in ('halton', 'hammersley', 'sobol'):
                skip = self._resample_count * n
                self._cached_pts = generate_collocation_pts(
                    self.sampling_method, n, seed=self.seed, skip=skip)
            else:
                self._cached_pts = generate_collocation_pts(
                    self.sampling_method, n, seed=self._resample_seed_counter)
                self._resample_seed_counter += 1

            self._resample_count += 1

        return self._cached_pts

    def train_step(self):
        # fixed grid path
        if self.sampling_method == 'grid' and self.resample_every is None:
            for batch in self.train_loader:
                x = batch[:, 0:1].requires_grad_(True)
                y = batch[:, 1:2].requires_grad_(True)
                residual = self.compute_residual(x, y)
                loss = self.loss_fn(residual, torch.zeros_like(residual))

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                self.loss_list.append(loss.item())
                self.loss = loss.item()
            return

        # general sampling path
        pts = self._get_collocation_pts()
        x = pts[:, 0:1].requires_grad_(True)
        y = pts[:, 1:2].requires_grad_(True)

        residual = self.compute_residual(x, y)
        loss = self.loss_fn(residual, torch.zeros_like(residual))

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.loss_list.append(loss.item())
        self.loss = loss.item()

    def train(self, plot_every=500):
        self.network.train(True)

        for self.epoch in range(self.epochs):
            if self.epoch % plot_every == 0 or self.epoch == self.epochs - 1:
                test_loss = self.compute_test_loss()
                self.test_loss_list.append(test_loss)
                self.test_epochs.append(self.epoch)

                tr, te = self.compute_true_errors()
                self.train_true_error_list.append(tr)
                self.test_true_error_list.append(te)
                self.true_error_epochs.append(self.epoch)

            self.train_step()


# ================================================================== #
#  Experiment runner
# ================================================================== #

# (kept minimal changes; comments added for clarity)

def run_experiment(label, sampling_method, resample_every,
                   num_hidden_nodes, num_layers, epochs, seeds,
                   x_range, y_range, results_dir,
                   num_collocation_pts=100, lr=1e-3, plot_every=500):

    exp_dir = os.path.join(results_dir, label)
    os.makedirs(exp_dir, exist_ok=True)

    summary = {
        'label': label,
        'sampling_method': sampling_method,
        'resample_every': resample_every,
        'final_test_errors': [],
        'final_train_losses': [],
        'final_test_losses': [],
        'times': [],
        'all_true_error_epochs': [],
        'all_train_true_error_lists': [],
        'all_test_true_error_lists': [],
        'all_loss_lists': [],
        'all_test_epochs': [],
        'all_test_loss_lists': [],
    }

    for s in seeds:
        print(f'\n--- {label} | seed {s} ---')
        torch.manual_seed(s)
        np.random.seed(s)
        random.seed(s)

        network = Fitter(num_hidden_nodes=num_hidden_nodes, num_layers=num_layers)
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(network.parameters(), lr=lr)

        train_set = DataSet(10, x_range, y_range)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))

        solver = PinnSolver(network, epochs, loss_fn, optimiser,
                            sampling_method, num_collocation_pts,
                            resample_every, x_range, y_range,
                            a=3, seed=s, train_loader=train_loader)

        t0 = time.time()
        solver.train(plot_every)
        elapsed = time.time() - t0

        print(f"Finished in {elapsed:.2f}s")

    return summary


# ================================================================== #
#  Main
# ================================================================== #

if __name__ == '__main__':
    print("Sampling experiment ready.")
