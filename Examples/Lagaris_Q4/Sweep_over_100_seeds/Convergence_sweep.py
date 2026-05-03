# -*- coding: utf-8 -*-
"""
100-seed PINN sweep with mode classification.
For each seed: trains with random collocation pts + curriculum,
records test loss & true loss, classifies mode, saves solution plot.

Run with:  python pinn_sweep.py
Outputs:   pinn_sweep_results/
              summary.png              - overview plots across all seeds
              contact_sheet.png        - all 100 seeds in a single grid (10x10)
              results.json             - all losses + mode per seed
              plots/seed_XXX_mode.png  - NN vs analytic per seed
"""

import torch
import torch.utils.data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, time
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Model classes
# ─────────────────────────────────────────────────────────────────────────────

class DataSet(torch.utils.data.Dataset):
    def __init__(self, num_samples, x_range):
        self.data_in   = torch.linspace(x_range[0], x_range[1], num_samples)
        self.x_range   = x_range
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, i):
        return self.data_in[i]


class Fitter(torch.nn.Module):
    def __init__(self, num_hidden_nodes):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, num_hidden_nodes)
        self.fc2 = torch.nn.Linear(num_hidden_nodes, 2)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h)


class PinnSolver:
    def __init__(self, network, train_set, epochs, train_loader,
                 loss_fn, optimiser, x_range=(0, 3), num_points=60, seed=0):
        self.network      = network
        self.train_set    = train_set
        self.train_loader = train_loader
        self.loss_fn      = loss_fn
        self.optimiser    = optimiser
        self.x_range      = x_range
        self.num_samples  = 10
        self.x_test       = torch.linspace(*x_range, num_points)
        self.seed         = seed
        self.epoch        = 0
        self.loss         = 0.
        self.epochs       = epochs
        self.xrespts      = 200
        self.xres_test    = torch.linspace(*x_range, self.xrespts)
        self.random_collocation_pts = True

        self.loss_list      = []
        self.test_loss_list = []

    # ── analytic / NN solutions ───────────────────────────────────────────────

    def analytic_solution(self):
        x = self.x_test.view(-1, 1)
        return torch.sin(x), 1 + x ** 2

    def nn_solution(self):
        x = self.x_test.view(-1, 1)
        N = self.network(x)
        psi1 = x * N[:, 0:1]
        psi2 = 1 + x * N[:, 1:2]
        return psi1, psi2

    # ── losses ────────────────────────────────────────────────────────────────

    def compute_test_loss(self):
        """Residual (physics) loss on a dense grid — does NOT require analytic soln."""
        self.network.eval()
        x = self.xres_test.detach().clone().view(-1, 1).requires_grad_(True)
        n_out = self.network(x)
        psi1  = x * n_out[:, 0:1]
        psi2  = 1 + x * n_out[:, 1:2]
        dpsi1 = torch.autograd.grad(psi1, x, grad_outputs=torch.ones_like(psi1), create_graph=True)[0]
        dpsi2 = torch.autograd.grad(psi2, x, grad_outputs=torch.ones_like(psi2), create_graph=False)[0]
        r1 = dpsi1 - (torch.cos(x) + psi1 ** 2 + psi2 - (1 + x ** 2 + torch.sin(x) ** 2))
        r2 = dpsi2 - (2 * x - (1 + x ** 2) * torch.sin(x) + psi1 * psi2)
        loss = self.loss_fn(r1, torch.zeros_like(r1)) + self.loss_fn(r2, torch.zeros_like(r2))
        self.network.train()
        return loss.item()

    def compute_true_loss(self):
        """MSE between NN solution and analytic solution."""
        self.network.eval()
        with torch.no_grad():
            psi1_nn,   psi2_nn   = self.nn_solution()
            psi1_true, psi2_true = self.analytic_solution()
            loss = self.loss_fn(psi1_nn, psi1_true) + self.loss_fn(psi2_nn, psi2_true)
        self.network.train()
        return loss.item()

    # ── training ──────────────────────────────────────────────────────────────

    def train_step(self):
        if self.random_collocation_pts:
            self.train_set.data_in = (
                self.x_range[0]
                + (self.x_range[1] - self.x_range[0]) * torch.rand(self.num_samples)
            )

        for batch in self.train_loader:
            batch = batch.view(-1, 1).requires_grad_(True)
            n_out = self.network(batch)
            x     = batch
            psi_1 = x * n_out[:, 0:1]
            psi_2 = 1 + x * n_out[:, 1:2]
            dpsi1_dx = torch.autograd.grad(psi_1, x, grad_outputs=torch.ones_like(psi_1), create_graph=True)[0]
            dpsi2_dx = torch.autograd.grad(psi_2, x, grad_outputs=torch.ones_like(psi_2), create_graph=True)[0]
            rhs1 = torch.cos(x) + psi_1 ** 2 + psi_2 - (1 + x ** 2 + torch.sin(x) ** 2)
            rhs2 = 2 * x - (1 + x ** 2) * torch.sin(x) + psi_1 * psi_2
            loss = (self.loss_fn(dpsi1_dx - rhs1, torch.zeros_like(dpsi1_dx)) +
                    self.loss_fn(dpsi2_dx - rhs2, torch.zeros_like(dpsi2_dx)))
            self.loss_list.append(loss.item())
            self.loss = loss.item()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    def train(self):
        self.network.train(True)
        for self.epoch in range(self.epochs):
            self.train_step()

    # ── per-seed plot ─────────────────────────────────────────────────────────

    def save_plot(self, path, test_loss, true_loss, mode_class):
        with torch.no_grad():
            psi1_nn,   psi2_nn   = self.nn_solution()
            psi1_true, psi2_true = self.analytic_solution()
            x         = self.x_test.view(-1, 1).numpy()
            psi1_nn   = psi1_nn.numpy()
            psi2_nn   = psi2_nn.numpy()
            psi1_true = psi1_true.numpy()
            psi2_true = psi2_true.numpy()

        MODE_COLORS = {
            "correct":   "#2ca02c",
            "failure_1": "#d62728",
            "failure_2": "#ff7f0e",
            "unknown":   "#9467bd",
        }
        border_color = MODE_COLORS.get(mode_class, "#aaa")

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(
            f"Seed {self.seed}  [{mode_class.upper()}]\n"
            f"Test Loss: {test_loss:.3e}  |  True Loss: {true_loss:.3e}",
            color="white", fontsize=11
        )

        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors='#ccc')
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        ax.plot(x, psi1_nn,   'r-',  lw=2,   label=r'NN $\hat\psi_1$')
        ax.plot(x, psi1_true, '--',  lw=1.5, color='#ffdd88', label=r'Exact $\psi_1$')
        ax.plot(x, psi2_nn,   'b-',  lw=2,   label=r'NN $\hat\psi_2$')
        ax.plot(x, psi2_true, '--',  lw=1.5, color='#88ffdd', label=r'Exact $\psi_2$')
        ax.set_title(r'$\psi_1$ and $\psi_2$', color='white')
        ax.legend(facecolor='#222', labelcolor='white', fontsize=8)
        ax.grid(True, color='#333')
        ax.set_xlabel('x', color='#ccc')

        plt.tight_layout()
        plt.savefig(path, dpi=90, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Mode classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_mode(true_loss, solver):
    """
    Three observed modes:
      correct   – true_loss small, NN tracks sin(x) and 1+x²
      failure_1 – psi1 collapsed (flat near zero)
      failure_2 – psi2 diverged / wildly wrong

    Adjust thresholds to match what you observe in your runs.
    """
    if true_loss < 0.05:
        return "correct"

    with torch.no_grad():
        psi1_nn, psi2_nn = solver.nn_solution()
        psi1 = psi1_nn.squeeze().numpy()
        psi2 = psi2_nn.squeeze().numpy()

    psi1_max   = float(np.max(np.abs(psi1)))
    psi2_range = float(np.max(psi2) - np.min(psi2))

    # psi1 flat / collapsed
    if psi1_max < 0.3:
        return "failure_1"

    # psi2 diverged or very large swing
    if psi2_range > 20 or true_loss > 5.0:
        return "failure_2"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Summary figure
# ─────────────────────────────────────────────────────────────────────────────

def make_contact_sheet(results, out_dir, ncols=10):
    """
    Single figure with all 100 seed plots arranged in a grid.
    Each cell shows ψ₁ and ψ₂ together (NN vs analytic), labelled with seed + mode.
    Border colour = mode colour.
    """
    MODE_COLOR = {
        "correct":   "#2ca02c",
        "failure_1": "#d62728",
        "failure_2": "#ff7f0e",
        "unknown":   "#9467bd",
    }

    n     = len(results)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.2, nrows * 1.8),
        squeeze=False
    )
    fig.patch.set_facecolor("#111118")

    for idx, r in enumerate(results):
        row  = idx // ncols
        col  = idx % ncols
        seed = r["seed"]
        mode = r["mode"]
        color = MODE_COLOR.get(mode, "#aaa")

        plot_path = os.path.join(out_dir, "plots", f"seed_{seed:03d}_{mode}.png")
        img = plt.imread(plot_path)

        ax = axes[row, col]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_facecolor("#111118")
        ax.set_title(f"s{seed}\n{mode}", fontsize=5.5, color=color, pad=2)

    # Hide any unused cells
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=m) for m, c in MODE_COLOR.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               facecolor='#222', labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("PINN 100-Seed Contact Sheet",
                 color="white", fontsize=13, y=1.002)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(out_dir, "contact_sheet.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Contact sheet saved → {out_path}")


def make_summary_figure(results, out_dir):
    seeds       = [r["seed"]      for r in results]
    test_losses = [r["test_loss"] for r in results]
    true_losses = [r["true_loss"] for r in results]
    modes       = [r["mode"]      for r in results]

    MODE_COLOR = {
        "correct":   "#2ca02c",
        "failure_1": "#d62728",
        "failure_2": "#ff7f0e",
        "unknown":   "#9467bd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("PINN 100-Seed Sweep — Summary", color="white", fontsize=15)

    for ax in axes.flat:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors='#ccc')
        ax.xaxis.label.set_color('#ccc')
        ax.yaxis.label.set_color('#ccc')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # 1. True loss per seed coloured by mode
    ax = axes[0, 0]
    for mode, color in MODE_COLOR.items():
        idx = [i for i, m in enumerate(modes) if m == mode]
        if idx:
            ax.scatter([seeds[i] for i in idx], [true_losses[i] for i in idx],
                       c=color, label=mode, s=30, alpha=0.85, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('Seed')
    ax.set_ylabel('True Loss (MSE vs analytic)')
    ax.set_title('True Loss per Seed')
    ax.legend(facecolor='#222', labelcolor='white', fontsize=9)
    ax.grid(True, color='#333')

    # 2. Test (residual) loss per seed
    ax = axes[0, 1]
    for mode, color in MODE_COLOR.items():
        idx = [i for i, m in enumerate(modes) if m == mode]
        if idx:
            ax.scatter([seeds[i] for i in idx], [test_losses[i] for i in idx],
                       c=color, label=mode, s=30, alpha=0.85, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Test Loss (residual MSE)')
    ax.set_title('Test Loss per Seed')
    ax.legend(facecolor='#222', labelcolor='white', fontsize=9)
    ax.grid(True, color='#333')

    # 3. Mode pie chart
    ax = axes[1, 0]
    counts = Counter(modes)
    labels = list(counts.keys())
    sizes  = [counts[l] for l in labels]
    colors = [MODE_COLOR.get(l, '#aaa') for l in labels]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.0f%%',
        textprops={'color': 'white'}, startangle=90
    )
    for at in autotexts:
        at.set_color('white')
    ax.set_title('Mode Distribution (n=100)')

    # 4. True loss histogram by mode
    ax = axes[1, 1]
    for mode, color in MODE_COLOR.items():
        vals = [true_losses[i] for i, m in enumerate(modes) if m == mode]
        if vals:
            log_vals = np.log10(np.clip(vals, 1e-10, None))
            ax.hist(log_vals, bins=20, color=color, alpha=0.75, label=mode)
    ax.set_xlabel('log₁₀(True Loss)')
    ax.set_ylabel('Count')
    ax.set_title('True Loss Distribution by Mode')
    ax.legend(facecolor='#222', labelcolor='white', fontsize=9)
    ax.grid(True, color='#333')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "summary.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Summary figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

CURRICULUM = [
    ((0, 3),  10000,  "stage-1 (0,3)"),
]

X_RANGE_FINAL = (0.0, 3.0)
NUM_POINTS    = 60


def run_sweep(num_seeds=3, out_dir="pinn_sweep_results"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    results = []
    t0 = time.time()

    for seed in range(num_seeds):
        # ── reproducibility ──────────────────────────────────────────────────
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        # ── build fresh network + optimiser ──────────────────────────────────
        network   = Fitter(num_hidden_nodes=10)
        loss_fn   = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(network.parameters(), lr=1e-3)

        train_set    = DataSet(num_samples=10, x_range=X_RANGE_FINAL)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=False)

        solver = PinnSolver(network, train_set, 0, train_loader,
                            loss_fn, optimiser, X_RANGE_FINAL, NUM_POINTS, seed)
        solver.random_collocation_pts = True

        # ── curriculum training ───────────────────────────────────────────────
        for x_r, ep, _ in CURRICULUM:
            solver.x_range      = x_r
            solver.epochs       = ep
            solver.train_set    = DataSet(num_samples=10, x_range=x_r)
            solver.train_loader = torch.utils.data.DataLoader(
                solver.train_set, batch_size=10, shuffle=False)
            solver.train()

        # ── evaluate on full range ────────────────────────────────────────────
        solver.x_range   = X_RANGE_FINAL
        solver.x_test    = torch.linspace(*X_RANGE_FINAL, NUM_POINTS)
        solver.xres_test = torch.linspace(*X_RANGE_FINAL, 200)

        test_loss = solver.compute_test_loss()
        true_loss = solver.compute_true_loss()
        mode      = classify_mode(true_loss, solver)

        # ── save per-seed plot ────────────────────────────────────────────────
        plot_path = os.path.join(out_dir, "plots", f"seed_{seed:03d}_{mode}.png")
        solver.save_plot(plot_path, test_loss, true_loss, mode)

        results.append({
            "seed":      seed,
            "test_loss": test_loss,
            "true_loss": true_loss,
            "mode":      mode,
        })

        elapsed = time.time() - t0
        eta     = elapsed / (seed + 1) * (num_seeds - seed - 1)
        print(f"[{seed+1:3d}/{num_seeds}] seed={seed:3d}  mode={mode:<10s}"
              f"  test={test_loss:.3e}  true={true_loss:.3e}"
              f"  ETA {eta/60:.1f} min")

    # ── save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── print mode summary ────────────────────────────────────────────────────
    counts = Counter(r["mode"] for r in results)
    print("\n=== MODE SUMMARY ===")
    for m, c in sorted(counts.items()):
        true_l = [r["true_loss"] for r in results if r["mode"] == m]
        print(f"  {m:<12s}: {c:3d} seeds ({100*c/num_seeds:.0f}%)"
              f"  median true_loss={np.median(true_l):.3e}")

    # ── summary figure ────────────────────────────────────────────────────────
    make_summary_figure(results, out_dir)

    # ── contact sheet ─────────────────────────────────────────────────────────
    print("\nBuilding contact sheet...")
    make_contact_sheet(results, out_dir)

    return results


if __name__ == "__main__":
    run_sweep(num_seeds=100)
