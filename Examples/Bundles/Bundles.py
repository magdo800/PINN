
"""
Section 5.1 in dissertation.
Solution Bundle for parametric ODE: y'(x) + lambda*y(x) = 0, y(0) = 1
Exact solution: y(x; lambda) = exp(-lambda * x)
Parameter range: lambda in [0.5, 3.0]
"""
 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
 
# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# ── Hyperparameters ───────────────────────────────────────────────────────────
N_x      = 20
N_lam    = 20
N_EPOCHS = 5000
LR       = 1e-3
LAM_LO, LAM_HI = 0.5, 3.0
X_LO,   X_HI   = 0.0, 1.0
IC_WEIGHT = 10.0
 
# Fixed held-out test grid (never used during training)
N_TEST_X   = 50
N_TEST_LAM = 50
TEST_X   = torch.linspace(X_LO,   X_HI,   N_TEST_X,   device=device)
TEST_LAM = torch.linspace(LAM_LO, LAM_HI, N_TEST_LAM, device=device)
_tx, _tl = torch.meshgrid(TEST_X, TEST_LAM, indexing="ij")
TEST_X_FLAT   = _tx.reshape(-1)
TEST_LAM_FLAT = _tl.reshape(-1)
 
 
# ── Network ───────────────────────────────────────────────────────────────────
class BundleNet(nn.Module):
    """
    Hard-constraint variant (Lagaris trick):
        f(x, λ) = 1 + x * g(x, λ)   →  f(0,λ) = 1 exactly for all λ
    """
    def __init__(self, hidden=64, depth=4, hard_ic=False):
        super().__init__()
        self.hard_ic = hard_ic
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
 
    def forward(self, x, lam):
        inp = torch.stack([x, lam], dim=-1)
        out = self.net(inp)
        if self.hard_ic:
            out = 1.0 + x.unsqueeze(-1) * out
        return out.squeeze(-1)
 
 
# ── Exact solution ────────────────────────────────────────────────────────────
def exact_np(x, lam):
    return np.exp(-lam * x)
 
 
# ── Losses ────────────────────────────────────────────────────────────────────
def residual_loss(model, x, lam, create_graph=True):
    x = x.requires_grad_(True)
    y = model(x, lam)
    dy = torch.autograd.grad(y.sum(), x, create_graph=create_graph)[0]
    return ((dy + lam * y) ** 2).mean()
 
 
def pinn_loss(model, x_flat, l_flat):
    loss_res = residual_loss(model, x_flat, l_flat, create_graph=True)
    if model.hard_ic:
        return loss_res
    lam_unique = l_flat[:N_lam]
    x0 = torch.zeros(N_lam, device=device)
    y0 = model(x0, lam_unique)
    loss_ic = ((y0 - 1.0) ** 2).mean()
    return loss_res + IC_WEIGHT * loss_ic
 
 
def compute_test_loss(model):
    """PDE residual on the fixed held-out (x, λ) grid."""
    with torch.enable_grad():
        loss = residual_loss(model,
                             TEST_X_FLAT.clone(),
                             TEST_LAM_FLAT.clone(),
                             create_graph=False)
    return loss.item()
 
 
# ── Training ──────────────────────────────────────────────────────────────────
def train(model, n_epochs=N_EPOCHS, lr=LR, test_every=50):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                            T_max=n_epochs)
    train_history = []
    test_history  = []
    test_epochs   = []
 
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()
 
        x_pts = torch.rand(N_x,   device=device) * (X_HI - X_LO) + X_LO
        l_pts = torch.rand(N_lam, device=device) * (LAM_HI - LAM_LO) + LAM_LO
        x_grid, l_grid = torch.meshgrid(x_pts, l_pts, indexing="ij")
        x_flat = x_grid.reshape(-1)
        l_flat = l_grid.reshape(-1)
 
        loss = pinn_loss(model, x_flat, l_flat)
        loss.backward()
        optimiser.step()
        scheduler.step()
 
        train_history.append(loss.item())
 
        if epoch % test_every == 0:
            model.eval()
            with torch.no_grad():
                tl = compute_test_loss(model)
            test_history.append(tl)
            test_epochs.append(epoch)
 
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:5d}/{n_epochs}  "
                  f"train={loss.item():.3e}  "
                  f"test={test_history[-1]:.3e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"t={time.time()-t0:.1f}s")
 
    return train_history, test_history, test_epochs
 
 
# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_model(model, x_np, lam_np):
    x   = torch.tensor(x_np,   dtype=torch.float32, device=device)
    lam = torch.tensor(lam_np, dtype=torch.float32, device=device)
    return model(x, lam).cpu().numpy()
 
 
# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_loss_curve(train_history, test_history, test_epochs, tag):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_all = np.arange(1, len(train_history) + 1)
    ax.semilogy(epochs_all, train_history,
                lw=1.0, color="#e05c2a", alpha=0.6, label="Train loss")
    ax.semilogy(test_epochs, test_history,
                lw=2.2, color="#3a7ebf", label="Test loss (held-out grid)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Train vs Test Loss — {tag.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"loss_curve_{tag}.png", dpi=150)
    plt.close()
    print(f"  Saved loss_curve_{tag}.png")
 
 
def plot_solution_quality(model, tag):
    x_fine   = np.linspace(X_LO, X_HI, 200, dtype=np.float32)
    lam_plot = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, lv in zip(axes.flatten(), lam_plot):
        x_t = torch.tensor(x_fine, device=device)
        l_t = torch.full_like(x_t, lv)
        with torch.no_grad():
            pred = model(x_t, l_t).cpu().numpy()
        ax.plot(x_fine, exact_np(x_fine, lv), "k--", lw=2, label="Exact")
        ax.plot(x_fine, pred, lw=2, color="#3a7ebf", label="Bundle NN")
        ax.set_title(f"λ = {lv}")
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Solution Bundle vs Exact — $y'+\\lambda y=0$ ({tag.replace('_',' ').title()})",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f"solution_quality_{tag}.png", dpi=150)
    plt.close()
    print(f"  Saved solution_quality_{tag}.png")
 
 
def plot_solution_heatmaps(model, tag):
    """Three-panel heatmap: exact | NN | error, shared (x, λ) axes."""
    x_fine   = np.linspace(X_LO, X_HI,   200, dtype=np.float32)
    lam_vals = np.linspace(LAM_LO, LAM_HI, 200, dtype=np.float32)
    X_g, L_g = np.meshgrid(x_fine, lam_vals, indexing="ij")
 
    pred_flat = eval_model(model, X_g.reshape(-1), L_g.reshape(-1))
    true_flat = exact_np(X_g.reshape(-1), L_g.reshape(-1))
 
    Z_true = true_flat.reshape(200, 200)
    Z_pred = pred_flat.reshape(200, 200)
    Z_err  = np.abs(Z_pred - Z_true)
 
    vmin_sol = min(Z_true.min(), Z_pred.min())
    vmax_sol = max(Z_true.max(), Z_pred.max())
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        (Z_true, "Exact  $e^{-\\lambda x}$",      "viridis", vmin_sol, vmax_sol),
        (Z_pred, "Bundle NN  $f(x,\\lambda)$",     "viridis", vmin_sol, vmax_sol),
        (Z_err,  "Absolute Error",                 "inferno", None,     None),
    ]
    for ax, (Z, title, cmap, vmin, vmax) in zip(axes, panels):
        kw = dict(cmap=cmap, shading="auto")
        if vmin is not None:
            kw.update(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(x_fine, lam_vals, Z.T, **kw)
        plt.colorbar(pcm, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("λ")
        ax.set_title(title)
 
    fig.suptitle(
        f"Solution Heatmaps — {tag.replace('_',' ').title()}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"solution_heatmaps_{tag}.png", dpi=150)
    plt.close()
    print(f"  Saved solution_heatmaps_{tag}.png")
    print(f"  MAE={Z_err.mean():.4e}   Max={Z_err.max():.4e}")
 
 
def plot_max_error_vs_lambda(model, tag):
    x_fine   = np.linspace(X_LO, X_HI,   200, dtype=np.float32)
    lam_vals = np.linspace(LAM_LO, LAM_HI, 200, dtype=np.float32)
    X_g, L_g = np.meshgrid(x_fine, lam_vals, indexing="ij")
    pred = eval_model(model, X_g.reshape(-1), L_g.reshape(-1))
    true = exact_np(X_g.reshape(-1), L_g.reshape(-1))
    err  = np.abs(pred - true).reshape(200, 200).max(axis=0)
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_vals, err, lw=2, color="#2aad6f")
    ax.set_xlabel("λ")
    ax.set_ylabel("max$_x$ |error|")
    ax.set_title(f"Worst-case Error vs λ — {tag.replace('_',' ').title()}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"max_error_vs_lambda_{tag}.png", dpi=150)
    plt.close()
    print(f"  Saved max_error_vs_lambda_{tag}.png")
 
 
def plot_extrapolation(models_dict, tag="extrapolation_soft_vs_hard"):
    """
    4 rows (λ = 3.5, 4.0, 4.5, 5.0) × 2 cols (Soft IC, Hard IC).
    Each subplot: exact vs NN with MAE annotated.
    """
    x_fine     = np.linspace(X_LO, X_HI, 200, dtype=np.float32)
    lam_extrap = [3.5, 4.0, 4.5, 5.0]
    names      = list(models_dict.keys())
 
    fig, axes = plt.subplots(len(lam_extrap), len(names),
                             figsize=(5 * len(names), 3.5 * len(lam_extrap)),
                             sharex=True)
 
    for row, lv in enumerate(lam_extrap):
        true = exact_np(x_fine, lv)
        for col, name in enumerate(names):
            ax = axes[row, col]
            m  = models_dict[name]
            x_t = torch.tensor(x_fine, device=device)
            l_t = torch.full_like(x_t, lv)
            with torch.no_grad():
                pred = m(x_t, l_t).cpu().numpy()
            mae = np.abs(pred - true).mean()
 
            ax.plot(x_fine, true, "k--", lw=2, label="Exact")
            ax.plot(x_fine, pred, lw=2,  color="#c0392b",
                    label=f"{name}  (MAE={mae:.2e})")
            ax.set_title(f"{name}  |  λ = {lv}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
 
    fig.suptitle(
        "Extrapolation: training range $\\lambda\\in[0.5,3]$, "
        "queried at $\\lambda\\in\\{3.5,4,4.5,5\\}$",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{tag}.png", dpi=150)
    plt.close()
    print(f"  Saved {tag}.png")
 
 
def make_all_figures(model, train_hist, test_hist, test_epochs, tag):
    plot_loss_curve(train_hist, test_hist, test_epochs, tag)
    plot_solution_quality(model, tag)
    plot_solution_heatmaps(model, tag)
    plot_max_error_vs_lambda(model, tag)
 
 
# ── Experiment: Soft IC vs Hard IC ───────────────────────────────────────────
def compare_hard_soft():
    models    = {}
    t_hists   = {}
    te_hists  = {}
    te_epochs = {}
 
    for name, hard in [("soft_ic", False), ("hard_ic", True)]:
        print(f"\n{'='*55}")
        print(f"  Training  {name}")
        print(f"{'='*55}")
        m = BundleNet(hidden=64, depth=4, hard_ic=hard).to(device)
        tr, te, tep = train(m)
        models[name]    = m
        t_hists[name]   = tr
        te_hists[name]  = te
        te_epochs[name] = tep
        make_all_figures(m, tr, te, tep, tag=name)
 
    # ── Overlaid loss curves (train + test) ───────────────────────────────────
    colours = {"soft_ic": "#e05c2a", "hard_ic": "#3a7ebf"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name in models:
        c  = colours[name]
        lb = name.replace("_", " ").title()
        axes[0].semilogy(t_hists[name], lw=1.0, color=c, alpha=0.6,
                         label=f"{lb} — train")
        axes[0].semilogy(te_epochs[name], te_hists[name], lw=2.2,
                         color=c, linestyle="--", label=f"{lb} — test")
        axes[1].semilogy(te_epochs[name], te_hists[name], lw=2.2,
                         color=c, label=f"{lb} — test")
    for ax, title in zip(axes, ["Train + Test Loss", "Test Loss Only"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Soft IC vs Hard IC — Loss Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig("loss_soft_vs_hard.png", dpi=150)
    plt.close()
    print("  Saved loss_soft_vs_hard.png")
 
    # ── Side-by-side error heatmaps (shared colour scale) ────────────────────
    x_fine   = np.linspace(X_LO, X_HI,   200, dtype=np.float32)
    lam_vals = np.linspace(LAM_LO, LAM_HI, 200, dtype=np.float32)
    X_g, L_g = np.meshgrid(x_fine, lam_vals, indexing="ij")
    true_flat = exact_np(X_g.reshape(-1), L_g.reshape(-1))
 
    errs = {}
    for name, m in models.items():
        pred = eval_model(m, X_g.reshape(-1), L_g.reshape(-1))
        errs[name] = np.abs(pred - true_flat).reshape(200, 200)
    vmax = max(e.max() for e in errs.values())
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, err) in zip(axes, errs.items()):
        pcm = ax.pcolormesh(x_fine, lam_vals, err.T,
                            cmap="inferno", shading="auto",
                            vmin=0, vmax=vmax)
        plt.colorbar(pcm, ax=ax, label="|error|")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("x")
        ax.set_ylabel("λ")
    fig.suptitle("Error Heatmap: Soft IC vs Hard IC  (shared colour scale)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("error_heatmap_comparison.png", dpi=150)
    plt.close()
    print("  Saved error_heatmap_comparison.png")
 
    # ── Extrapolation: soft vs hard, λ = 3.5 / 4 / 4.5 / 5 ──────────────────
    plot_extrapolation(
        {"Soft IC": models["soft_ic"], "Hard IC": models["hard_ic"]},
        tag="extrapolation_soft_vs_hard",
    )
 
    return models
 
 
# ── Sample efficiency sweep ───────────────────────────────────────────────────
def sample_efficiency_experiment():
    ns = [5, 10, 20, 40]
    results = {}
    x_eval = np.linspace(X_LO, X_HI,   100, dtype=np.float32)
    l_eval = np.linspace(LAM_LO, LAM_HI, 100, dtype=np.float32)
    X_e, L_e = np.meshgrid(x_eval, l_eval, indexing="ij")
    true_flat = exact_np(X_e.reshape(-1), L_e.reshape(-1))
 
    for n in ns:
        print(f"\n── Sample efficiency N={n} ──")
        global N_x, N_lam
        N_x = N_lam = n
        m = BundleNet(hidden=64, depth=4, hard_ic=False).to(device)
        train(m, n_epochs=3000)
        pred = eval_model(m, X_e.reshape(-1), L_e.reshape(-1))
        mae  = np.abs(pred - true_flat).mean()
        results[n] = mae
        print(f"  N={n}  MAE={mae:.4e}")
 
    N_x = N_lam = 20  # reset
 
    ns_arr  = np.array(ns)
    mae_arr = np.array([results[n] for n in ns])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(ns_arr, mae_arr, "o-", lw=2, color="#9b59b6")
    ax.set_xlabel("N (= N_x = N_λ)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Sample Efficiency: Error vs Collocation Count")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("sample_efficiency.png", dpi=150)
    plt.close()
    print("  Saved sample_efficiency.png")
    return results
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
 
    print("=" * 55)
    print("  EXPERIMENTS 1 & 2: Soft IC vs Hard IC")
    print("=" * 55)
    compare_hard_soft()
 
    print("\n" + "=" * 55)
    print("  EXPERIMENT 3: Sample efficiency sweep")
    print("=" * 55)
    sample_efficiency_experiment()
 
    print("\nAll done. Figures saved to current directory.")
