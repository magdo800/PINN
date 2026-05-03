# -*- coding: utf-8 -*-
"""
Sampling Method Experiment
==========================
Compares collocation-point sampling strategies for a 2-D PINN in two regimes:

  A) Fixed   – points drawn once at the start, never changed.
  B) Resampling – points redrawn every `RESAMPLE_EVERY` epochs.

In both regimes the **fixed uniform grid** is used as a control.

Sampling methods
----------------
  grid        Equispaced uniform grid  (deterministic control)
  random      Pseudo-random uniform    (torch.rand / PCG-64)
  lhs         Latin Hypercube Sampling (scipy.stats.qmc)
  halton      Halton low-discrepancy sequence
  hammersley  Hammersley sequence
  sobol       Sobol base-2 sequence

Plots produced  (saved to RESULTS_DIR/comparison_*.png)
--------------------------------------------------------
  1. True-error curves – Fixed setting     (mean ± geo-std band, log scale)
  2. True-error curves – Resampling setting
  3. Side-by-side panel combining 1 & 2
  4. Final-error grouped bar chart (Fixed vs Resampling per method)

Per-seed text files
-------------------
  RESULTS_DIR/<label>/seed_<s>/metrics.txt
      final train loss, final test loss, final true test error, elapsed time
"""

import torch
import torch.utils.data
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    Return a (n, 2) float32 tensor of collocation points on [0,1]^2.

    Parameters
    ----------
    method : one of 'grid', 'random', 'lhs', 'halton', 'hammersley', 'sobol'
    n      : number of points (for 'grid', the nearest perfect square is used)
    seed   : RNG seed (used for stochastic methods and scipy qmc)
    skip   : number of sequence points to skip before drawing n points.
             Used by quasi-random methods (Halton, Hammersley, Sobol) so that
             each resample draws a genuinely different, non-overlapping block
             from the low-discrepancy sequence.  Ignored by grid/random/lhs.
    """
    if method == 'grid':
        # equispaced grid – take ceil(sqrt(n)) per side
        side = int(np.ceil(np.sqrt(n)))
        xs = torch.linspace(0., 1., side)
        ys = torch.linspace(0., 1., side)
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')
        pts = torch.stack([gx.flatten(), gy.flatten()], dim=1).float()
        # trim to exactly n if needed
        return pts[:n]

    elif method == 'random':
        rng = torch.Generator()
        rng.manual_seed(seed)
        return torch.rand(n, 2, generator=rng)

    elif method == 'lhs':
        # LHS has no natural "skip"; use seed to get a fresh independent draw
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        samples = sampler.random(n)           # shape (n,2), in [0,1]^2
        return torch.tensor(samples, dtype=torch.float32)

    elif method == 'halton':
        # Advance the sequence by `skip` points, then draw the next n.
        # This gives non-overlapping blocks on every resample.
        sampler = qmc.Halton(d=2, scramble=False)
        sampler.fast_forward(skip)
        samples = sampler.random(n)
        return torch.tensor(samples, dtype=torch.float32)

    elif method == 'hammersley':
        # Hammersley: first dim equidistant within the block, second dim = Halton.
        # Advance the 1-D Halton component by `skip`, then draw n points.
        halton_sampler = qmc.Halton(d=1, scramble=False)
        halton_sampler.fast_forward(skip)
        halton = halton_sampler.random(n)                    # (n,1)
        # Re-index the equidistant dimension relative to the current block
        first = np.arange(1, n + 1).reshape(-1, 1) / n  # (n,1) — always spans [0,1]
        samples = np.hstack([first, halton])                 # (n,2)
        return torch.tensor(samples, dtype=torch.float32)

    elif method == 'sobol':
        # Sobol requires draws at powers of 2.  Skip `skip` points then draw n.
        # We over-draw to the next power of 2 and trim.
        m = int(np.ceil(np.log2(max(skip + n, 1))))
        sampler = qmc.Sobol(d=2, scramble=False)
        all_pts = sampler.random_base2(m)         # shape (2^m, 2)
        samples = all_pts[skip:skip + n]
        if len(samples) < n:
            # Fallback: if skip+n exceeds 2^m wrap around (shouldn't normally happen)
            samples = all_pts[:n]
        return torch.tensor(samples, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown sampling method: '{method}'")


# ================================================================== #
#  Dataset  (used for the fixed-grid baseline train_loader path)
# ================================================================== #

class DataSet(torch.utils.data.Dataset):
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
#  Network
# ================================================================== #

class Fitter(torch.nn.Module):
    def __init__(self, num_hidden_nodes, num_layers=1):
        super().__init__()
        assert num_layers >= 1
        layers = [torch.nn.Linear(2, num_hidden_nodes), torch.nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [torch.nn.Linear(num_hidden_nodes, num_hidden_nodes), torch.nn.Tanh()]
        layers.append(torch.nn.Linear(num_hidden_nodes, 1))
        self.net = torch.nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


# ================================================================== #
#  Solver
# ================================================================== #

class PinnSolver:
    def __init__(self, network, epochs, loss_fn, optimiser,
                 sampling_method='grid', num_collocation_pts=100,
                 resample_every=None,
                 x_range=(0, 1), y_range=(0, 1),
                 a=3, seed=0, plot_samples=30,
                 train_loader=None):
        """
        Parameters
        ----------
        sampling_method     : str  – one of the six methods above
        num_collocation_pts : int  – target number of collocation points
        resample_every      : int or None
                              None  → fixed mode (sample once, never resample)
                              int   → resample every this many epochs
        train_loader        : only used when sampling_method == 'grid' AND
                              resample_every is None (the pure fixed-grid control)
        """
        self.network             = network
        self.epochs              = epochs
        self.loss_fn             = loss_fn
        self.optimiser           = optimiser
        self.sampling_method     = sampling_method
        self.num_collocation_pts = num_collocation_pts
        self.resample_every      = resample_every   # None → fixed
        self.x_range             = x_range
        self.y_range             = y_range
        self.a                   = a
        self.seed                = seed
        self.plot_samples        = plot_samples
        self.train_loader        = train_loader  # fallback for grid-control path

        self._cached_pts  = None
        self._resample_seed_counter = seed  # increments on each resample (random/lhs)
        self._resample_count = 0            # how many times we've resampled (quasi-random skip)

        self.epoch          = 0
        self.loss           = 0.
        self.loss_list      = []
        self.test_loss_list = []
        self.test_epochs    = []

        self.train_true_error_list = []
        self.test_true_error_list  = []
        self.true_error_epochs     = []

        # test grid (plot_samples x plot_samples)
        n  = self.plot_samples
        xs = torch.linspace(*self.x_range, n)
        ys = torch.linspace(*self.y_range, n)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        self._test_xf = X.flatten().unsqueeze(1)
        self._test_yf = Y.flatten().unsqueeze(1)

        # train reference grid (10×10) for true-error evaluation
        xst = torch.linspace(*self.x_range, 10)
        yst = torch.linspace(*self.y_range, 10)
        Xt, Yt = torch.meshgrid(xst, yst, indexing='ij')
        self._train_xf = Xt.flatten().unsqueeze(1)
        self._train_yf = Yt.flatten().unsqueeze(1)

    # ------------------------------------------------------------------ #
    #  Exact solution & boundary term
    # ------------------------------------------------------------------ #

    def psi_exact(self, x, y):
        a = self.a
        return torch.exp(-(a * x + y) / 5) * torch.sin(a**2 * x**2 + y)

    def A(self, x, y):
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
        xy = torch.cat([x, y], dim=1)
        N  = self.network(xy)
        return self.A(x, y) + x * (1 - x) * y * (1 - y) * N

    def compute_residual(self, x, y):
        a   = self.a
        psi = self.psi_trial(x, y)
        dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),
                                      create_graph=True)[0]
        dpsi_dy = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi),
                                      create_graph=True)[0]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx),
                                        create_graph=True)[0]
        d2psi_dy2 = torch.autograd.grad(dpsi_dy, y, grad_outputs=torch.ones_like(dpsi_dy),
                                        create_graph=True)[0]
        laplacian = d2psi_dx2 + d2psi_dy2
        f = torch.exp(-(a * x + y) / 5) * (
              ((-4 / 5) * a**3 * x - 2 / 5 + 2 * a**2) * torch.cos(a**2 * x**2 + y)
            + (1 / 25 - 1 - 4 * a**4 * x**2 + a**2 / 25) * torch.sin(a**2 * x**2 + y)
        )
        return laplacian - f

    # ------------------------------------------------------------------ #
    #  Evaluation helpers
    # ------------------------------------------------------------------ #

    def reset_adam_moments(self, optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                optimizer.state[p] = {}

    def compute_test_loss(self):
        self.network.eval()
        xf = self._test_xf.detach().clone().requires_grad_(True)
        yf = self._test_yf.detach().clone().requires_grad_(True)
        residual = self.compute_residual(xf, yf)
        loss = self.loss_fn(residual, torch.zeros_like(residual))
        self.network.train()
        return loss.item()

    def compute_true_errors(self):
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

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def _get_collocation_pts(self):
        """
        Return collocation points according to sampling_method and resample schedule.

        For quasi-random sequences (Halton, Hammersley, Sobol), each resample
        advances `skip` by n points so that every draw is a new, non-overlapping
        block from the low-discrepancy sequence — giving genuine variation.

        For stochastic methods (random, lhs), a different seed is used each time.

        For the fixed-grid control (method='grid', resample_every=None) this
        method is not called — the train_loader is used instead.
        """
        need_resample = (
            self._cached_pts is None
            or (self.resample_every is not None
                and self.epoch % self.resample_every == 0)
        )

        if need_resample:
            n = self.num_collocation_pts
            self.reset_adam_moments(self.optimiser)
            if self.sampling_method in ('halton', 'hammersley', 'sobol'):
                # Advance through the sequence: block k starts at k*n
                skip = self._resample_count * n
                self._cached_pts = generate_collocation_pts(
                    self.sampling_method, n, seed=self.seed, skip=skip)
            else:
                # random / lhs: use an incrementing seed for independent draws
                s = self._resample_seed_counter
                self._cached_pts = generate_collocation_pts(
                    self.sampling_method, n, seed=s, skip=0)
                self._resample_seed_counter += 1

            self._resample_count += 1

        return self._cached_pts

    def train_step(self):
        # ---- pure fixed-grid control: use the DataLoader ----
        if self.sampling_method == 'grid' and self.resample_every is None:
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
            return

        # ---- all other methods (fixed or resampling) ----
        pts = self._get_collocation_pts()
        x   = pts[:, 0:1].detach().requires_grad_(True)
        y   = pts[:, 1:2].detach().requires_grad_(True)

        residual = self.compute_residual(x, y)
        loss     = self.loss_fn(residual, torch.zeros_like(residual))

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

def run_experiment(label, sampling_method, resample_every,
                   num_hidden_nodes, num_layers, epochs, seeds,
                   x_range, y_range, results_dir,
                   num_collocation_pts=100, lr=1e-3, plot_every=500):
    """
    Train one configuration (method × regime) over multiple seeds.

    resample_every=None  → fixed regime
    resample_every=int   → resampling regime
    """
    exp_dir = os.path.join(results_dir, label)
    os.makedirs(exp_dir, exist_ok=True)

    summary = {
        'label':            label,
        'sampling_method':  sampling_method,
        'resample_every':   resample_every,
        'final_test_errors':  [],
        'final_train_losses': [],
        'final_test_losses':  [],
        'times':              [],
        'all_true_error_epochs':      [],
        'all_train_true_error_lists': [],
        'all_test_true_error_lists':  [],
        'all_loss_lists':             [],
        'all_test_epochs':            [],
        'all_test_loss_lists':        [],
    }

    for s in seeds:
        print(f'\n--- {label} | seed {s} ---')
        torch.manual_seed(s)
        np.random.seed(s)
        random.seed(s)

        network   = Fitter(num_hidden_nodes=num_hidden_nodes, num_layers=num_layers)
        loss_fn   = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(network.parameters(), lr=lr)

        # build a train_loader for the fixed-grid control path
        train_set    = DataSet(num_samples=10, x_range=x_range, y_range=y_range)
        train_loader = torch.utils.data.DataLoader(
                           train_set, batch_size=len(train_set), shuffle=False)

        solver = PinnSolver(
            network             = network,
            epochs              = epochs,
            loss_fn             = loss_fn,
            optimiser           = optimiser,
            sampling_method     = sampling_method,
            num_collocation_pts = num_collocation_pts,
            resample_every      = resample_every,
            x_range             = x_range,
            y_range             = y_range,
            a                   = 3,
            seed                = s,
            plot_samples        = 30,
            train_loader        = train_loader,
        )

        t0      = time.time()
        solver.train(plot_every=plot_every)
        elapsed = time.time() - t0

        # ---- per-seed npz files ----
        seed_dir = os.path.join(exp_dir, f'seed_{s}')
        os.makedirs(seed_dir, exist_ok=True)

        np.savez(os.path.join(seed_dir, 'true_error_vs_epoch.npz'),
                 true_error_epochs     = np.array(solver.true_error_epochs),
                 train_true_error_list = np.array(solver.train_true_error_list),
                 test_true_error_list  = np.array(solver.test_true_error_list))

        np.savez(os.path.join(seed_dir, 'loss_vs_epoch.npz'),
                 loss_list      = np.array(solver.loss_list),
                 test_epochs    = np.array(solver.test_epochs),
                 test_loss_list = np.array(solver.test_loss_list))

        # ---- per-seed metrics text file ----
        with open(os.path.join(seed_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Configuration:       {label}\n")
            f.write(f"Sampling method:     {sampling_method}\n")
            f.write(f"Resample every:      {resample_every if resample_every is not None else 'N/A (fixed)'}\n")
            f.write(f"Seed:                {s}\n")
            f.write(f"Num collocation pts: {num_collocation_pts}\n")
            f.write(f"Epochs:              {epochs}\n")
            f.write(f"Learning rate:       {lr}\n")
            f.write(f"Hidden nodes:        {num_hidden_nodes}\n")
            f.write(f"Num layers:          {num_layers}\n")
            f.write(f"Elapsed time (s):    {elapsed:.2f}\n")
            f.write(f"Final train loss:    {solver.loss:.6e}\n")
            f.write(f"Final test loss:     {solver.test_loss_list[-1]:.6e}\n")
            f.write(f"Final true test err: {solver.test_true_error_list[-1]:.6e}\n")
            f.write(f"Final true train err:{solver.train_true_error_list[-1]:.6e}\n")

        # ---- accumulate summary ----
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

        print(f'  Finished in {elapsed:.1f}s | True test err = {solver.test_true_error_list[-1]:.4e}')

    return summary


# ================================================================== #
#  Plotting helpers
# ================================================================== #

# colour palette: one colour per method, consistent across all plots
METHOD_COLORS = {
    'grid':        '#444444',
    'random':      '#E07B39',
    'lhs':         '#4C8BE2',
    'halton':      '#2BAE66',
    'hammersley':  '#C642A0',
    'sobol':       '#E8B84B',
}

METHOD_LABELS = {
    'grid':        'Grid (control)',
    'random':      'Random',
    'lhs':         'LHS',
    'halton':      'Halton',
    'hammersley':  'Hammersley',
    'sobol':       'Sobol',
}


def _geo_mean_std_band(values_2d):
    """
    Given a 2-D array (seeds × epochs), return:
        mean_curve, upper_band, lower_band
    all on the original (non-log) scale, computed geometrically.
    """
    arr   = np.array(values_2d, dtype=float)
    log_a = np.log10(arr)
    m     = log_a.mean(axis=0)
    s     = log_a.std(axis=0)
    mean  = 10 ** m
    upper = 10 ** (m + s)
    lower = 10 ** (m - s)
    return mean, upper, lower


def _geo_mean_std_scalar(values):
    log_v = np.log10(np.array(values, dtype=float))
    m = log_v.mean()
    s = log_v.std()
    mean = 10 ** m
    up   = 10 ** (m + s) - mean
    dn   = mean - 10 ** (m - s)
    return mean, up, dn


def _plot_error_curves_panel(ax, summaries, title, key='all_test_true_error_lists'):
    """
    Plot mean true-error curves with geometric-std shading onto `ax`.
    `summaries` is a list of summary dicts for one regime.
    """
    for s in summaries:
        method = s['sampling_method']
        colour = METHOD_COLORS.get(method, 'grey')
        label  = METHOD_LABELS.get(method, method)

        ep           = np.array(s['all_true_error_epochs'][0])
        mean, up, lo = _geo_mean_std_band(s[key])

        ls = '--' if method == 'grid' else '-'
        ax.plot(ep, mean, color=colour, linewidth=2.0, linestyle=ls, label=label, zorder=3)
        ax.fill_between(ep, lo, up, color=colour, alpha=0.15, zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Relative L2 Error', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)


def plot_error_curves_fixed(summaries_fixed, results_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_error_curves_panel(ax, summaries_fixed,
                             'True Test Error – Fixed Collocation Points')
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_fixed_error_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_error_curves_resample(summaries_resample, results_dir, resample_every):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_error_curves_panel(ax, summaries_resample,
                             f'True Test Error – Resampling Every {resample_every} Epochs')
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_resample_error_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_error_curves_sidebyside(summaries_fixed, summaries_resample,
                                  results_dir, resample_every):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    _plot_error_curves_panel(axes[0], summaries_fixed,
                             'Fixed Collocation Points')
    _plot_error_curves_panel(axes[1], summaries_resample,
                             f'Resampling Every {resample_every} Epochs')
    fig.suptitle('True Test Error — Fixed vs Resampling\n'
                 '(geometric mean ± std over seeds)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_sidebyside_error_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_final_error_grouped_bar(summaries_fixed, summaries_resample, results_dir):
    """
    Grouped bar chart: for each sampling method, one bar for Fixed and one for Resampling.
    """
    # build aligned lists
    methods = [s['sampling_method'] for s in summaries_fixed]

    fixed_means, fixed_ups, fixed_dns = [], [], []
    res_means,   res_ups,   res_dns   = [], [], []

    for sf, sr in zip(summaries_fixed, summaries_resample):
        m, u, d = _geo_mean_std_scalar(sf['final_test_errors'])
        fixed_means.append(m); fixed_ups.append(u); fixed_dns.append(d)
        m, u, d = _geo_mean_std_scalar(sr['final_test_errors'])
        res_means.append(m); res_ups.append(u); res_dns.append(d)

    x     = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, method in enumerate(methods):
        col = METHOD_COLORS.get(method, 'grey')
        ax.bar(x[i] - width/2, fixed_means[i], width,
               color=col, alpha=0.95, label='Fixed' if i == 0 else '',
               edgecolor='black', linewidth=0.6, zorder=2)
        ax.bar(x[i] + width/2, res_means[i], width,
               color=col, alpha=0.50, label='Resampling' if i == 0 else '',
               edgecolor='black', linewidth=0.6, hatch='//', zorder=2)

    ax.errorbar(x - width/2, fixed_means,
                yerr=[fixed_dns, fixed_ups],
                fmt='none', color='black', capsize=4, linewidth=1.2, zorder=3)
    ax.errorbar(x + width/2, res_means,
                yerr=[res_dns, res_ups],
                fmt='none', color='black', capsize=4, linewidth=1.2, zorder=3)

    tick_labels = [METHOD_LABELS.get(m, m) for m in methods]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yscale('log')
    ax.set_ylabel('Final Relative L2 Error (geometric mean ± std)', fontsize=11)
    ax.set_title('Final True Test Error — Fixed vs Resampling per Sampling Method', fontsize=12)

    # custom legend: solid = fixed, hatched = resampling
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='grey', edgecolor='black', label='Fixed'),
        Patch(facecolor='grey', edgecolor='black', hatch='//', alpha=0.5, label='Resampling'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_final_error_grouped_bar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ================================================================== #
#  Main
# ================================================================== #

if __name__ == '__main__':

    # ---- global hyperparameters ----
    SEEDS            = [0, 1, 2]
    EPOCHS           = 50000
    LR               = 1e-3
    PLOT_EVERY       = 500
    NUM_HIDDEN_NODES = 10
    NUM_LAYERS       = 1
    NUM_COLLOCATION  = 100
    X_RANGE          = (0., 1.)
    Y_RANGE          = (0., 1.)
    RESAMPLE_EVERY   = 1000    # used for the resampling regime
    RESULTS_DIR      = 'results_sampling_experiment'

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- sampling methods to test ----
    METHODS = ['grid', 'random', 'lhs', 'halton', 'hammersley', 'sobol']

    all_summaries_fixed    = []
    all_summaries_resample = []

    # ================================================================
    #  A) Fixed regime
    # ================================================================
    print('\n' + '='*70)
    print('  REGIME A: Fixed collocation points')
    print('='*70)

    for method in METHODS:
        label = f'fixed_{method}'
        print(f'\n{"="*60}\n  {label}\n{"="*60}')
        summary = run_experiment(
            label               = label,
            sampling_method     = method,
            resample_every      = None,          # ← fixed
            num_hidden_nodes    = NUM_HIDDEN_NODES,
            num_layers          = NUM_LAYERS,
            epochs              = EPOCHS,
            seeds               = SEEDS,
            x_range             = X_RANGE,
            y_range             = Y_RANGE,
            results_dir         = RESULTS_DIR,
            num_collocation_pts = NUM_COLLOCATION,
            lr                  = LR,
            plot_every          = PLOT_EVERY,
        )
        all_summaries_fixed.append(summary)

    # ================================================================
    #  B) Resampling regime
    # ================================================================
    print('\n' + '='*70)
    print(f'  REGIME B: Resampling every {RESAMPLE_EVERY} epochs')
    print('='*70)

    for method in METHODS:
        label = f'resample{RESAMPLE_EVERY}_{method}'
        print(f'\n{"="*60}\n  {label}\n{"="*60}')
        summary = run_experiment(
            label               = label,
            sampling_method     = method,
            resample_every      = RESAMPLE_EVERY,  # ← resampling
            num_hidden_nodes    = NUM_HIDDEN_NODES,
            num_layers          = NUM_LAYERS,
            epochs              = EPOCHS,
            seeds               = SEEDS,
            x_range             = X_RANGE,
            y_range             = Y_RANGE,
            results_dir         = RESULTS_DIR,
            num_collocation_pts = NUM_COLLOCATION,
            lr                  = LR,
            plot_every          = PLOT_EVERY,
        )
        all_summaries_resample.append(summary)

    # ---- save all summaries ----
    np.save(os.path.join(RESULTS_DIR, 'all_summaries_fixed.npy'),
            all_summaries_fixed, allow_pickle=True)
    np.save(os.path.join(RESULTS_DIR, 'all_summaries_resample.npy'),
            all_summaries_resample, allow_pickle=True)
    print(f'\nSaved summary arrays.')

    # ================================================================
    #  Plots
    # ================================================================
    plot_error_curves_fixed(all_summaries_fixed, RESULTS_DIR)
    plot_error_curves_resample(all_summaries_resample, RESULTS_DIR, RESAMPLE_EVERY)
    plot_error_curves_sidebyside(all_summaries_fixed, all_summaries_resample,
                                  RESULTS_DIR, RESAMPLE_EVERY)
    plot_final_error_grouped_bar(all_summaries_fixed, all_summaries_resample,
                                  RESULTS_DIR)

    print('\nAll done.')
