# -*- coding: utf-8 -*-
"""
Resampling Frequency Experiment
================================
Compares a fixed collocation grid baseline against random resampling at
various frequencies: every 1, 10, 50, 100, 250, 500, and 1000 epochs.

Plots produced
--------------
1. Final true error across all configurations       (bar chart)
2. Final train loss and test loss                   (grouped bar chart)
3. True test error curves over training             (all configs, same axes)
4. True train error curves over training            (all configs, same axes)

Per-configuration data saved
-----------------------------
  results_dir/<label>/seed_<s>/
      true_error_vs_epoch.npz   – arrays: true_error_epochs,
                                           train_true_error_list,
                                           test_true_error_list
      loss_vs_epoch.npz         – arrays: loss_list (train, per epoch),
                                           test_epochs,
                                           test_loss_list
      metrics.txt

All comparison plots go to results_dir/comparison_*.png
All summaries saved to results_dir/all_summaries.npy (reload with allow_pickle=True)
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

class PinnSolver():
    def __init__(self, network, train_set, epochs, train_loader, loss_fn, optimiser,
                 x_range=(0, 1), y_range=(0, 1), a=3, seed=0, plot_samples=30):
        self.network      = network
        self.train_set    = train_set
        self.epochs       = epochs
        self.train_loader = train_loader
        self.loss_fn      = loss_fn
        self.optimiser    = optimiser
        self.x_range      = x_range
        self.y_range      = y_range
        self.a            = a
        self.seed         = seed
        self.plot_samples = plot_samples

        self.random_collocation_pts = False
        self.num_collocation_pts    = self.train_set.num_samples ** 2
        self.resample_every         = 1      # only used when random_collocation_pts=True
        self._cached_pts            = None   # cache for infrequent resampling

        self.epoch          = 0
        self.loss           = 0.
        self.loss_list      = []
        self.test_loss_list = []
        self.test_epochs    = []

        self.train_true_error_list = []
        self.test_true_error_list  = []
        self.true_error_epochs     = []

        # test grid (30x30)
        n = self.plot_samples
        xs = torch.linspace(*self.x_range, n)
        ys = torch.linspace(*self.y_range, n)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        self._test_xf = X.flatten().unsqueeze(1)
        self._test_yf = Y.flatten().unsqueeze(1)

        # train grid (num_samples x num_samples)
        nt  = self.train_set.num_samples
        xst = torch.linspace(*self.x_range, nt)
        yst = torch.linspace(*self.y_range, nt)
        Xt, Yt = torch.meshgrid(xst, yst, indexing='ij')
        self._train_xf = Xt.flatten().unsqueeze(1)
        self._train_yf = Yt.flatten().unsqueeze(1)

    # ------------------------------------------------------------------ #
    #  Exact solution and boundary term
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

    def train_step(self):
        if self.random_collocation_pts:
            # resample on scheduled epochs, or if cache is still empty
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

def run_resample_experiment(label, random_pts, resample_every,
                             num_hidden_nodes, num_layers, epochs, seeds,
                             x_range, y_range, results_dir,
                             num_collocation_pts=100, lr=1e-3, plot_every=500):
    """
    Train one configuration across multiple seeds. Returns a summary dict
    containing per-seed curves and final scalars.

    Parameters
    ----------
    random_pts     : bool  – False = fixed grid baseline
    resample_every : int   – how often to draw new points; ignored if random_pts=False
    """
    exp_dir = os.path.join(results_dir, label)
    os.makedirs(exp_dir, exist_ok=True)

    summary = {
        'label':              label,
        'random_pts':         random_pts,
        'resample_every':     resample_every,
        'num_hidden_nodes':   num_hidden_nodes,
        'num_layers':         num_layers,
        'epochs':             epochs,
        # --- final scalars (one per seed) ---
        'final_test_errors':  [],
        'final_train_losses': [],
        'final_test_losses':  [],
        'times':              [],
        # --- full curves (one list per seed) ---
        #     true_error_epochs is the same for every seed (given same plot_every)
        #     but stored per seed for safety
        'all_true_error_epochs':      [],
        'all_train_true_error_lists': [],
        'all_test_true_error_lists':  [],
        'all_loss_lists':             [],   # per-epoch train loss
        'all_test_epochs':            [],
        'all_test_loss_lists':        [],
    }

    for s in seeds:
        print(f'\n--- {label} | seed {s} ---')
        torch.manual_seed(s); np.random.seed(s); random.seed(s)

        network      = Fitter(num_hidden_nodes=num_hidden_nodes, num_layers=num_layers)
        train_set    = DataSet(num_samples=10, x_range=x_range, y_range=y_range)
        train_loader = torch.utils.data.DataLoader(
                           train_set, batch_size=len(train_set), shuffle=False)
        loss_fn   = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(network.parameters(), lr=lr)

        solver = PinnSolver(network, train_set, epochs, train_loader, loss_fn, optimiser,
                            x_range=x_range, y_range=y_range, a=3,
                            seed=s, plot_samples=30)
        solver.random_collocation_pts = random_pts
        solver.num_collocation_pts    = num_collocation_pts
        if random_pts:
            solver.resample_every = resample_every

        t0 = time.time()
        solver.train(plot_every=plot_every)
        elapsed = time.time() - t0

        # --- save per-seed curves as .npz (reload with np.load(...)) ---
        seed_dir = os.path.join(exp_dir, f'seed_{s}')
        os.makedirs(seed_dir, exist_ok=True)

        np.savez(os.path.join(seed_dir, 'true_error_vs_epoch.npz'),
                 true_error_epochs      = np.array(solver.true_error_epochs),
                 train_true_error_list  = np.array(solver.train_true_error_list),
                 test_true_error_list   = np.array(solver.test_true_error_list))

        np.savez(os.path.join(seed_dir, 'loss_vs_epoch.npz'),
                 loss_list       = np.array(solver.loss_list),
                 test_epochs     = np.array(solver.test_epochs),
                 test_loss_list  = np.array(solver.test_loss_list))

        with open(os.path.join(seed_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Label:               {label}\n")
            f.write(f"Seed:                {s}\n")
            f.write(f"Random pts:          {random_pts}\n")
            f.write(f"Resample every:      {resample_every}\n")
            f.write(f"Num collocation pts: {num_collocation_pts}\n")
            f.write(f"Elapsed time (s):    {elapsed:.4f}\n")
            f.write(f"Final train loss:    {solver.loss:.6e}\n")
            f.write(f"Final test loss:     {solver.test_loss_list[-1]:.6e}\n")
            f.write(f"Final true test err: {solver.test_true_error_list[-1]:.6e}\n")

        # --- accumulate into summary ---
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
#  Comparison plots
# ================================================================== #

def _geo_mean_std(values):
    """Geometric mean + asymmetric std bands suitable for log-scale plots."""
    log_v = np.log10(np.array(values, dtype=float))
    m = np.mean(log_v)
    s = np.std(log_v)
    mean = 10**m
    up   = 10**(m + s) - mean
    dn   = mean - 10**(m - s)
    return mean, up, dn


def plot_final_true_errors(summaries, results_dir):
    """Bar chart: mean final relative L2 test error per configuration."""
    labels = [s['label'] for s in summaries]
    means, ups, dns = [], [], []
    for s in summaries:
        m, u, d = _geo_mean_std(s['final_test_errors'])
        means.append(m); ups.append(u); dns.append(d)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, color='steelblue', alpha=0.85, zorder=2)
    ax.errorbar(x, means, yerr=[dns, ups], fmt='none', color='black',
                capsize=5, linewidth=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Mean final relative L2 error (geometric ± std over seeds)')
    ax.set_title('Final True Error — all configurations')
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_final_true_error.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_final_losses(summaries, results_dir):
    """Grouped bar chart: final train loss and final test (PDE residual) loss."""
    labels = [s['label'] for s in summaries]
    x = np.arange(len(labels))
    width = 0.38

    def geo_bars(lists):
        ms, us, ds = [], [], []
        for vals in lists:
            m, u, d = _geo_mean_std(vals)
            ms.append(m); us.append(u); ds.append(d)
        return ms, us, ds

    tr_m, tr_u, tr_d = geo_bars([s['final_train_losses'] for s in summaries])
    te_m, te_u, te_d = geo_bars([s['final_test_losses']  for s in summaries])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, tr_m, width, label='Train loss', color='tomato',    alpha=0.85, zorder=2)
    ax.bar(x + width/2, te_m, width, label='Test loss',  color='steelblue', alpha=0.85, zorder=2)
    ax.errorbar(x - width/2, tr_m, yerr=[tr_d, tr_u], fmt='none',
                color='black', capsize=4, linewidth=1.2, zorder=3)
    ax.errorbar(x + width/2, te_m, yerr=[te_d, te_u], fmt='none',
                color='black', capsize=4, linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Mean final loss (geometric ± std over seeds)')
    ax.set_title('Final Train & Test (PDE residual) Loss — all configurations')
    ax.legend(); ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_final_losses.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_true_error_curves(summaries, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    panel_info = [
        ('all_test_true_error_lists',  'True Error — Test Grid (30×30)'),
        ('all_train_true_error_lists', 'True Error — Train Points (10×10)'),
    ]
    cmap = plt.cm.get_cmap('tab10', len(summaries))

    for ax, (key, title) in zip(axes, panel_info):
        for idx, s in enumerate(summaries):
            colour     = cmap(idx)
            ep         = np.array(s['all_true_error_epochs'][0])
            arr        = np.array(s[key])
            mean_curve = arr.mean(axis=0)

            ax.plot(ep, mean_curve, color=colour, linewidth=1.8, label=s['label'])

        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative L2 Error')
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle('True Error Curves — all configurations\n'
                 '(mean over seeds)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_true_error_curves.png')
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
    PLOT_EVERY       = 500        # snapshot interval for curves and test loss
    NUM_HIDDEN_NODES = 10
    NUM_LAYERS       = 1
    NUM_COLLOCATION  = 100        # number of random pts when resampling
    X_RANGE          = (0., 1.)
    Y_RANGE          = (0., 1.)
    RESULTS_DIR      = 'results_resample_experiment'

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- configurations: (label, random_pts, resample_every) ----
    # resample_every is ignored for the fixed grid baseline
    configs = [
        ('fixed_grid',          False, None),
        ('resample_every_1000',    True,     100),
        ('resample_every_2000',   True,    200),
        ('resample_every_3000',   True,    300),
        ('resample_every_4000',  True,   500),
        ('resample_every_5000',  True,   1000),
    ]

    all_summaries = []

    for label, random_pts, resample_every in configs:
        print(f'\n{"="*60}')
        print(f'  Configuration: {label}')
        print(f'{"="*60}')
        summary = run_resample_experiment(
            label               = label,
            random_pts          = random_pts,
            resample_every      = resample_every if resample_every is not None else 1,
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
        all_summaries.append(summary)

    # ---- save all summaries for later analysis ----
    # reload with: summaries = np.load('all_summaries.npy', allow_pickle=True).tolist()
    np.save(os.path.join(RESULTS_DIR, 'all_summaries.npy'),
            all_summaries, allow_pickle=True)
    print(f'\nSaved all_summaries.npy  ({len(all_summaries)} configurations)')

    # ---- comparison plots ----
    plot_final_true_errors(all_summaries, RESULTS_DIR)
    plot_final_losses(all_summaries, RESULTS_DIR)
    plot_true_error_curves(all_summaries, RESULTS_DIR)

    print('\nAll done.')
