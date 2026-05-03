
"""
LagQ6: Resampling frequency experiment for Lagaris Problem 6, section 3.2 in dissertation.

Resampling Frequency Experiment
================================
Compares a fixed collocation grid baseline against random resampling at
various frequencies: every 1, 10, 50, 100, 250, 500, and 1000 epochs.

"""

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os


# ================================================================== #
#  Dataset
# ================================================================== #

class DataSet(torch.utils.data.Dataset):
    """
    Generates a structured 2D grid dataset over a rectangular domain.

    Each sample is a coordinate pair (x, y) flattened from a meshgrid.
    """
    def __init__(self, num_samples, x_range, y_range):
        # Create 1D linspaces for each axis
        x = torch.linspace(x_range[0], x_range[1], num_samples)
        y = torch.linspace(y_range[0], y_range[1], num_samples)

        # Build 2D grid from coordinate vectors
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # Flatten grid into N x 2 input tensor
        self.data_in = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        self.x_range, self.y_range = x_range, y_range
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, i):
        return self.data_in[i]


# ================================================================== #
#  Neural Network (Fitter)
# ================================================================== #

class Fitter(torch.nn.Module):
    """
    Fully connected feedforward neural network used as trial function.

    Architecture:
        input (x,y) -> hidden layers (tanh) -> output scalar
    """
    def __init__(self, num_hidden_nodes, num_layers=1):
        super().__init__()
        assert num_layers >= 1

        # Build sequential MLP
        layers = [torch.nn.Linear(2, num_hidden_nodes), torch.nn.Tanh()]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers += [torch.nn.Linear(num_hidden_nodes, num_hidden_nodes), torch.nn.Tanh()]

        # Output layer (scalar field)
        layers.append(torch.nn.Linear(num_hidden_nodes, 1))

        self.net = torch.nn.Sequential(*layers)

        # Xavier initialization for stable training
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


# ================================================================== #
#  PINN Solver
# ================================================================== #

class PinnSolver():
    """
    Physics-Informed Neural Network solver.

    Responsibilities:
        - Construct trial solution satisfying boundary conditions
        - Compute PDE residual via autograd
        - Train network on collocation points
        - Track training + true error metrics
    """
    def __init__(self, network, train_set, epochs, train_loader, loss_fn, optimiser,
                 x_range=(0, 1), y_range=(0, 1), a=3, seed=0, plot_samples=30):

        # Core components
        self.network      = network
        self.train_set    = train_set
        self.epochs       = epochs
        self.train_loader = train_loader
        self.loss_fn      = loss_fn
        self.optimiser    = optimiser

        # Domain
        self.x_range      = x_range
        self.y_range      = y_range

        # PDE parameter
        self.a            = a

        # Reproducibility tracking
        self.seed         = seed

        # Resolution for evaluation grids
        self.plot_samples = plot_samples

        # Collocation configuration
        self.random_collocation_pts = False
        self.num_collocation_pts    = self.train_set.num_samples ** 2
        self.resample_every         = 1
        self._cached_pts            = None

        # Training state
        self.epoch          = 0
        self.loss           = 0.
        self.loss_list      = []
        self.test_loss_list = []
        self.test_epochs    = []

        # True error tracking
        self.train_true_error_list = []
        self.test_true_error_list  = []
        self.true_error_epochs     = []

        # ========================================================== #
        # Precompute evaluation grids
        # ========================================================== #

        # Test grid (coarse)
        n = self.plot_samples
        xs = torch.linspace(*self.x_range, n)
        ys = torch.linspace(*self.y_range, n)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        self._test_xf = X.flatten().unsqueeze(1)
        self._test_yf = Y.flatten().unsqueeze(1)

        # Train grid (dense)
        nt  = self.train_set.num_samples
        xst = torch.linspace(*self.x_range, nt)
        yst = torch.linspace(*self.y_range, nt)
        Xt, Yt = torch.meshgrid(xst, yst, indexing='ij')
        self._train_xf = Xt.flatten().unsqueeze(1)
        self._train_yf = Yt.flatten().unsqueeze(1)

    # ========================================================== #
    #  Exact solution & boundary construction
    # ========================================================== #

    def psi_exact(self, x, y):
        """Analytical reference solution."""
        a = self.a
        return torch.exp(-(a * x + y) / 5) * torch.sin(a**2 * x**2 + y)

    def A(self, x, y):
        """
        Boundary correction term ensuring trial solution satisfies BCs.
        """
        A_val = (
            (1 - x) * self.psi_exact(torch.zeros_like(x), y)
          +       x  * self.psi_exact(torch.ones_like(x),  y)
          + (1 - y) * (self.psi_exact(x, torch.zeros_like(y))
                       - (1 - x) * self.psi_exact(torch.zeros_like(x), torch.zeros_like(y))
                       -       x  * self.psi_exact(torch.ones_like(x),  torch.zeros_like(y)))
          +       y  * (self.psi_exact(x, torch.ones_like(y))
                       - (1 - x) * self.psi_exact(torch.zeros_like(x), torch.ones_like(y))
                       -       x  * self.psi_exact(torch.ones_like(x),  torch.ones_like(y)))
        )
        return A_val

    def psi_trial(self, x, y):
        """
        Trial solution: satisfies boundary conditions by construction.
        """
        xy = torch.cat([x, y], dim=1)
        N  = self.network(xy)
        return self.A(x, y) + x * (1 - x) * y * (1 - y) * N

    def compute_residual(self, x, y):
        """
        PDE residual computed via automatic differentiation.
        """
        a   = self.a
        psi = self.psi_trial(x, y)

        # First derivatives
        dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),
                                      create_graph=True)[0]
        dpsi_dy = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi),
                                      create_graph=True)[0]

        # Second derivatives
        d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx),
                                        create_graph=True)[0]
        d2psi_dy2 = torch.autograd.grad(dpsi_dy, y, grad_outputs=torch.ones_like(dpsi_dy),
                                        create_graph=True)[0]

        laplacian = d2psi_dx2 + d2psi_dy2

        # Forcing term (PDE RHS)
        f = torch.exp(-(a * x + y) / 5) * (
              ((-4 / 5) * a**3 * x - 2 / 5 + 2 * a**2) * torch.cos(a**2 * x**2 + y)
            + (1 / 25 - 1 - 4 * a**4 * x**2 + a**2 / 25) * torch.sin(a**2 * x**2 + y)
        )

        return laplacian - f

    def compute_test_loss(self):
        """PDE residual loss on test grid."""
        self.network.eval()

        xf = self._test_xf.detach().clone().requires_grad_(True)
        yf = self._test_yf.detach().clone().requires_grad_(True)

        residual = self.compute_residual(xf, yf)
        loss = self.loss_fn(residual, torch.zeros_like(residual))

        self.network.train()
        return loss.item()

    def compute_true_errors(self):
        """
        Relative L2 error vs analytical solution on train/test grids.
        """
        self.network.eval()
        with torch.no_grad():
            psi_tr    = self.psi_trial(self._train_xf, self._train_yf)
            psi_ex_tr = self.psi_exact(self._train_xf, self._train_yf)
            train_err = (torch.norm(psi_tr - psi_ex_tr) / torch.norm(psi_ex_tr)).item()

            psi_te    = self.psi_trial(self._test_xf, self._test_yf)
            psi_ex_te = self.psi_exact(self._test_xf, self._test_yf)
            test_err  = (torch.norm(psi_te - psi_ex_te) / torch.norm(psi_ex_te)).item()

        self.network.train()
        return train_err, test_err

    # ========================================================== #
    #  Training loop
    # ========================================================== #

    def train_step(self):
        """Single optimizer step depending on collocation strategy."""

        if self.random_collocation_pts:
            # Random resampling strategy
            if self._cached_pts is None or self.epoch % self.resample_every == 0:
                self._cached_pts = torch.rand(self.num_collocation_pts, 2)

            pts = self._cached_pts
            x = pts[:, 0:1].requires_grad_(True)
            y = pts[:, 1:2].requires_grad_(True)

            residual = self.compute_residual(x, y)
            loss     = self.loss_fn(residual, torch.zeros_like(residual))

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            self.loss_list.append(loss.item())
            self.loss = loss.item()

        else:
            # Fixed grid training
            for batch in self.train_loader:
                x = batch[:, 0:1].requires_grad_(True)
                y = batch[:, 1:2].requires_grad_(True)

                residual = self.compute_residual(x, y)
                loss     = self.loss_fn(residual, torch.zeros_like(residual))

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                self.loss_list.append(loss.item())
                self.loss = loss.item()

    def train(self, plot_every=500):
        """Main training loop with periodic evaluation logging."""
        self.network.train(True)

        for self.epoch in range(self.epochs):

            # Periodic evaluation block
            if self.epoch % plot_every == 0 or self.epoch == self.epochs - 1:
                test_loss = self.compute_test_loss()
                self.test_loss_list.append(test_loss)
                self.test_epochs.append(self.epoch)

                tr_err, te_err = self.compute_true_errors()
                self.train_true_error_list.append(tr_err)
                self.test_true_error_list.append(te_err)
                self.true_error_epochs.append(self.epoch)

                print(f'  Epoch {self.epoch:5d} | Train loss {self.loss:.4e} '
                      f'| Test loss {test_loss:.4e} | True test err {te_err:.4e}')

            self.train_step()

        print(f'  Final train loss = {self.loss:.6e}')


# ================================================================== #
#  Experiment runner
# ================================================================== #

# NOTE: This runs multiple seeds for statistical stability

def run_resample_experiment(label, random_pts, resample_every,
                             num_hidden_nodes, num_layers, epochs, seeds,
                             x_range, y_range, results_dir,
                             num_collocation_pts=100, lr=1e-3, plot_every=500):

    exp_dir = os.path.join(results_dir, label)
    os.makedirs(exp_dir, exist_ok=True)

    summary = {
        # configuration metadata
        'label': label,
        'random_pts': random_pts,
        'resample_every': resample_every,

        # hyperparameters
        'num_hidden_nodes': num_hidden_nodes,
        'num_layers': num_layers,
        'epochs': epochs,

        # aggregated results
        'final_test_errors': [],
        'final_train_losses': [],
        'final_test_losses': [],
        'times': [],

        # full training curves
        'all_true_error_epochs': [],
        'all_train_true_error_lists': [],
        'all_test_true_error_lists': [],
        'all_loss_lists': [],
        'all_test_epochs': [],
        'all_test_loss_lists': [],
    }

    # ========================================================== #
    #  Loop over random seeds
    # ========================================================== #

    for s in seeds:
        print(f'\n--- {label} | seed {s} ---')

        # Ensure reproducibility
        torch.manual_seed(s)
        np.random.seed(s)
        random.seed(s)

        # Model + data
        network      = Fitter(num_hidden_nodes=num_hidden_nodes, num_layers=num_layers)
        train_set    = DataSet(num_samples=10, x_range=x_range, y_range=y_range)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)

        loss_fn   = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(network.parameters(), lr=lr)

        solver = PinnSolver(network, train_set, epochs, train_loader, loss_fn, optimiser,
                            x_range=x_range, y_range=y_range, a=3,
                            seed=s, plot_samples=30)

        # Configure collocation strategy
        solver.random_collocation_pts = random_pts
        solver.num_collocation_pts    = num_collocation_pts
        if random_pts:
            solver.resample_every = resample_every

        # Train timing
        t0 = time.time()
        solver.train(plot_every=plot_every)
        elapsed = time.time() - t0

        # Save per-seed outputs
        seed_dir = os.path.join(exp_dir, f'seed_{s}')
        os.makedirs(seed_dir, exist_ok=True)

        np.savez(os.path.join(seed_dir, 'true_error_vs_epoch.npz'),
                 true_error_epochs=solver.true_error_epochs,
                 train_true_error_list=solver.train_true_error_list,
                 test_true_error_list=solver.test_true_error_list)

        np.savez(os.path.join(seed_dir, 'loss_vs_epoch.npz'),
                 loss_list=solver.loss_list,
                 test_epochs=solver.test_epochs,
                 test_loss_list=solver.test_loss_list)

        # Simple text summary for quick inspection
        with open(os.path.join(seed_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Label: {label}\n")
            f.write(f"Seed: {s}\n")
            f.write(f"Elapsed: {elapsed:.4f}s\n")

        # Aggregate
        summary['final_test_errors'].append(solver.test_true_error_list[-1])
        summary['final_train_losses'].append(solver.loss)
        summary['final_test_losses'].append(solver.test_loss_list[-1])
        summary['times'].append(elapsed)

        summary['all_true_error_epochs'].append(list(solver.true_error_epochs))
        summary['all_train_true_error_lists'].append(list(solver.train_true_error_list))
        summary['all_test_true_error_lists'].append(list(solver.test_true_error_list))
        summary['all_loss_lists'].append(list(solver.loss_list))
        summary['all_test_epochs'].append(list(solver.test_epochs))
        summary['all_test_loss_lists'].append(list(solver.test_loss_list))

        print(f'  Finished in {elapsed:.1f}s')

    return summary


# ================================================================== #
#  Plot utilities
# ================================================================== #

# NOTE: plots compare configurations across seeds

# (Remaining plotting functions unchanged except comments omitted for brevity)

# ================================================================== #
#  Main execution
# ================================================================== #

if __name__ == '__main__':
    # Global experiment settings
    SEEDS = [0, 1, 2]
    EPOCHS = 50000
    LR = 1e-3

    # Frequency study configurations
    configs = [
        ('fixed_grid', False, None),
    
        # Resampling frequencies matching experiment description
        ('resample_every_1',    True, 1),
        ('resample_every_10',   True, 10),
        ('resample_every_50',   True, 50),
        ('resample_every_100',  True, 100),
        ('resample_every_250',  True, 250),
        ('resample_every_500',  True, 500),
        ('resample_every_1000', True, 1000),
    ]

    print("Experiment setup complete. Ready to run.")
