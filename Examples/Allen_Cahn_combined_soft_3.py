"""
Allen-Cahn PINN — pure PyTorch
Vanilla MLP vs Fourier Feature MLP

PDE:  u_t - d*u_xx - 5*(u - u^3) = 0,  x in [-1,1], t in [0,1]
IC:   u(x,0) = x^2 * cos(pi*x)
BC:   u(-1,t) = u(1,t) = -1

Hyper-parameters follow Wang et al. (2022), arXiv:2111.02801
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ── 0. Device ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 1. Config ─────────────────────────────────────────────────────────────────

CFG = dict(
    d            = 0.0001,       # ε² in the PDE
    layers       = 1,            # hidden layers
    width        = 10,          # neurons per hidden layer
    sigma        = 2.0,          # Fourier feature scale
    num_fourier  = 256,          # M  →  2M inputs after cos/sin
    lr           = 1e-3,
    decay_rate   = 1,
    decay_steps  = 5000,
    iterations   = 50_000,
    batch_domain = 8000,
    batch_bc     = 400,
    batch_ic     = 800,
    log_every    = 500,
    seed         = 22,
)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ── 2. Reference data ─────────────────────────────────────────────────────────

def load_reference():
    data = loadmat("/home3/lgml42/NCC_TEST/Allen_Cahn_features/dataset/Allen_Cahn.mat")
    x, t, u = data["x"], data["t"], data["u"]          # u shape: (nt, nx)
    xx, tt  = np.meshgrid(x, t)
    X = np.column_stack([xx.ravel(), tt.ravel()]).astype(np.float32)
    y = u.ravel()[:, None].astype(np.float32)
    return X, y                                         # (N,2), (N,1)

X_ref, y_ref = load_reference()
X_ref_t = torch.tensor(X_ref, device=device)
y_ref_t = torch.tensor(y_ref, device=device)


# ── 3. Collocation point samplers ─────────────────────────────────────────────

def sample_domain(n):
    """Uniform random points in [-1,1] x [0,1], requires_grad for autograd."""
    x = torch.empty(n, 2, device=device)
    x[:, 0].uniform_(-1.0, 1.0)   # spatial
    x[:, 1].uniform_(0.0,  1.0)   # temporal
    return x.requires_grad_(True)

def sample_ic(n):
    """t=0 points."""
    x = torch.empty(n, 2, device=device)
    x[:, 0].uniform_(-1.0, 1.0)
    x[:, 1].fill_(0.0)
    return x.requires_grad_(True)

def sample_bc(n):
    """x=-1 and x=+1 points, half each."""
    half = n // 2
    x = torch.empty(n, 2, device=device)
    x[:half,  0].fill_(-1.0)
    x[half:,  0].fill_( 1.0)
    x[:, 1].uniform_(0.0, 1.0)
    return x.requires_grad_(True)


# ── 4. Hard-constraint output transform ───────────────────────────────────────
#
#   u(x,t) = x^2 cos(pi x) + t*(1 - x^2)*(net(x,t) + 1)
#
#   This bakes in IC and BCs exactly, so we only need the PDE residual loss.
#   Set USE_HARD = False to use soft penalty losses instead.

USE_HARD = False     # match original script default

def hard_transform(xy, raw):
    """xy: (N,2) with grad,  raw: (N,1) network output."""
    x, t = xy[:, 0:1], xy[:, 1:2]
    return x**2 * torch.cos(np.pi * x) + t * (1 - x**2) * (raw + 1)


# ── 5. PDE residual ───────────────────────────────────────────────────────────

def pde_residual(model, xy):
    """
    Returns the Allen-Cahn residual:
        R = u_t - d*u_xx - 5*(u - u^3)
    Uses autograd; xy must have requires_grad=True.
    """
    u = model(xy)

    u_x  = torch.autograd.grad(u, xy, torch.ones_like(u),
                                create_graph=True)[0][:, 0:1]
    u_t  = torch.autograd.grad(u, xy, torch.ones_like(u),
                                create_graph=True)[0][:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x),
                                create_graph=True)[0][:, 0:1]

    return u_t - CFG["d"] * u_xx - 5.0 * (u - u**3)


# ── 6. Networks ───────────────────────────────────────────────────────────────

def make_mlp(in_dim, width, n_layers, out_dim=1):
    """Standard MLP with Glorot (Xavier) uniform init."""
    layers = []
    dims   = [in_dim] + [width] * n_layers + [out_dim]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        lin = nn.Linear(a, b)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class VanillaPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_mlp(2, CFG["width"], CFG["layers"])

    def forward(self, xy):
        raw = self.net(xy)
        return hard_transform(xy, raw) if USE_HARD else raw


class FourierPINN(nn.Module):
    """
    Random Fourier feature embedding followed by the same MLP backbone.

    B ~ N(0, σ²I) is fixed (registered as a buffer, not a parameter).
    Embedding: z = [cos(2π B x), sin(2π B x)]  →  shape (N, 2M)
    """
    def __init__(self):
        super().__init__()
        M     = CFG["num_fourier"]
        sigma = CFG["sigma"]
        B     = torch.randn(2, M) * sigma
        self.register_buffer("B", B)
        self.net = make_mlp(2 * M, CFG["width"], CFG["layers"])

    def forward(self, xy):
        proj = 2.0 * np.pi * (xy @ self.B)             # (N, M)
        z    = torch.cat([torch.cos(proj),
                          torch.sin(proj)], dim=-1)     # (N, 2M)
        raw  = self.net(z)
        return hard_transform(xy, raw) if USE_HARD else raw


# ── 7. IC / BC loss helpers ───────────────────────────────────────────────────

def ic_loss(model, n):
    xy   = sample_ic(n)
    x    = xy[:, 0:1]
    u    = model(xy)
    u_ic = x**2 * torch.cos(np.pi * x)
    return nn.functional.mse_loss(u, u_ic)

def bc_loss(model, n):
    xy  = sample_bc(n)
    u   = model(xy)
    return nn.functional.mse_loss(u, -torch.ones_like(u))


# ── 8. L2 relative error ──────────────────────────────────────────────────────

@torch.no_grad()
def l2_relative_error(model):
    u_pred = model(X_ref_t)
    num    = torch.norm(y_ref_t - u_pred)
    den    = torch.norm(y_ref_t)
    return (num / den).item()


# ── 9. Training loop ──────────────────────────────────────────────────────────

def train(model, label=""):
    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    # True exponential decay: lr * decay_rate^(step / decay_steps)
    # PyTorch's LambdaLR multiplies the base lr by the returned factor.
    gamma = CFG["decay_rate"] ** (1.0 / CFG["decay_steps"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    # ExponentialLR multiplies lr by gamma every step() call.
    # Since we call scheduler.step() once per iteration:
    #   lr(t) = lr_0 * gamma^t = lr_0 * decay_rate^(t / decay_steps)  ✓

    history = {
        "steps":      [],
        "loss_pde":   [],
        "loss_ic":    [],
        "loss_bc":    [],
        "loss_total": [],
        "l2":         [],
    }

    for step in range(1, CFG["iterations"] + 1):

        opt.zero_grad()

        # ── Resample collocation points every iteration ──────────────────
        xy_pde = sample_domain(CFG["batch_domain"])

        # ── Losses ──────────────────────────────────────────────────────
        res        = pde_residual(model, xy_pde)
        loss_pde   = (res**2).mean()

        if USE_HARD:
            loss_ic  = torch.tensor(0.0, device=device)
            loss_bc  = torch.tensor(0.0, device=device)
        else:
            loss_ic  = ic_loss(model, CFG["batch_ic"])
            loss_bc  = bc_loss(model, CFG["batch_bc"])

        loss = loss_pde + loss_ic + loss_bc

        loss.backward()
        opt.step()
        scheduler.step()

        # ── Logging ──────────────────────────────────────────────────────
        if step % CFG["log_every"] == 0:
            model.eval()
            l2 = l2_relative_error(model)
            model.train()

            current_lr = scheduler.get_last_lr()[0]
            history["steps"].append(step)
            history["loss_pde"].append(loss_pde.item())
            history["loss_ic"].append(loss_ic.item())
            history["loss_bc"].append(loss_bc.item())
            history["loss_total"].append(loss.item())
            history["l2"].append(l2)

            if step % 10_000 == 0:
                print(f"[{label}] step {step:>7d} | "
                      f"loss {loss.item():.3e} | "
                      f"pde {loss_pde.item():.3e} | "
                      f"ic {loss_ic.item():.3e} | "
                      f"bc {loss_bc.item():.3e} | "
                      f"L2 {l2:.4f} | "
                      f"lr {current_lr:.2e}")

    return history


# ── 10. Plotting ──────────────────────────────────────────────────────────────

def reshape_grid(X, y, nx=201, nt=101):
    x_vals = np.unique(X[:, 0])
    t_vals = np.unique(X[:, 1])
    return x_vals, t_vals, y.reshape(nt, nx)


def plot_loss_and_l2(history, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(history["steps"], history["loss_total"], label="Total",  lw=2)
    ax.semilogy(history["steps"], history["loss_pde"],   label="PDE",    lw=1.5, ls="--")
    if not USE_HARD:
        ax.semilogy(history["steps"], history["loss_ic"], label="IC",    lw=1.5, ls=":")
        ax.semilogy(history["steps"], history["loss_bc"], label="BC",    lw=1.5, ls="-.")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss (log)")
    ax.set_title("Training loss"); ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogy(history["steps"], history["l2"], "o-", color="tomato",
                lw=2, ms=3, label="L2 relative error")
    ax.set_xlabel("Iteration"); ax.set_ylabel("L2 (log)")
    ax.set_title("True error vs iteration")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(f"Training diagnostics  ({label})", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"loss_and_l2_{label}.png", dpi=150)
    plt.show()


def plot_solution(y_pred_np, label="", nx=201, nt=101):
    x_vals, t_vals, U_true = reshape_grid(X_ref, y_ref, nx, nt)
    _,      _,      U_pred = reshape_grid(X_ref, y_pred_np, nx, nt)
    U_err = np.abs(U_true - U_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (U, title, cmap, vmin, vmax) in zip(axes, [
        (U_true, "True solution",   "RdBu_r", -1, 1),
        (U_pred, "PINN prediction", "RdBu_r", -1, 1),
        (U_err,  "Absolute error",  "hot_r",   0, None),
    ]):
        im = ax.pcolormesh(x_vals, t_vals, U, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(title)

    plt.suptitle(f"Allen-Cahn solution  ({label})", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"solution_{label}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_slices(y_pred_np, label="", t_slices=(0.25, 0.5, 0.75, 1.0), nx=201, nt=101):
    x_vals, t_vals, U_true = reshape_grid(X_ref, y_ref, nx, nt)
    _,      _,      U_pred = reshape_grid(X_ref, y_pred_np, nx, nt)

    fig, axes = plt.subplots(1, len(t_slices), figsize=(14, 3.5), sharey=True)
    for ax, t_target in zip(axes, t_slices):
        idx = np.argmin(np.abs(t_vals - t_target))
        ax.plot(x_vals, U_true[idx], "k-",  lw=2,   label="True")
        ax.plot(x_vals, U_pred[idx], "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {t_vals[idx]:.2f}")
        ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("u(x,t)"); ax.legend()

    plt.suptitle(f"Solution slices  ({label})", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"slices_{label}.png", dpi=150)
    plt.show()


def plot_comparison(hist_van, hist_fou, sigma, l2_van, l2_fou):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(hist_van["steps"], hist_van["l2"],
                "o-", lw=2, ms=3, label="Vanilla PINN")
    ax.semilogy(hist_fou["steps"], hist_fou["l2"],
                "s-", lw=2, ms=3, label=f"Fourier PINN (σ={sigma})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("L2 relative error (log)")
    ax.set_title("Vanilla vs Fourier feature PINN")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_vanilla_vs_fourier.png", dpi=150)
    plt.show()
    print(f"\nFinal L2 — Vanilla: {l2_van:.6f}  |  Fourier: {l2_fou:.6f}")


@torch.no_grad()
def predict(model):
    model.eval()
    return model(X_ref_t).cpu().numpy()




# ── 11. Run ───────────────────────────────────────────────────────────────────

# Vanilla
print("=" * 60)
print("VANILLA PINN  (4 × 256, tanh, Adam + exp decay)")
print("=" * 60)

vanilla_model   = VanillaPINN()
vanilla_history = train(vanilla_model, label="Vanilla")
y_pred_van      = predict(vanilla_model)
l2_van          = l2_relative_error(vanilla_model)
res_van         = np.abs(y_pred_van - y_ref)

print(f"\nVanilla — mean residual: {res_van.mean():.6f}")
print(f"Vanilla — L2 relative error: {l2_van:.6f}")
np.savetxt("test_vanilla.dat", np.hstack([X_ref, y_ref, y_pred_van]))

plot_loss_and_l2(vanilla_history, label="Vanilla")
plot_solution(y_pred_van,         label="Vanilla")
plot_slices(y_pred_van,           label="Vanilla")


# Fourier
print("=" * 60)
print(f"FOURIER PINN  (σ={CFG['sigma']}, M={CFG['num_fourier']}, 4×256, tanh)")
print("=" * 60)

fourier_model   = FourierPINN()
fourier_history = train(fourier_model, label=f"Fourier_sigma{CFG['sigma']}")
y_pred_fou      = predict(fourier_model)
l2_fou          = l2_relative_error(fourier_model)
res_fou         = np.abs(y_pred_fou - y_ref)

print(f"\nFourier — mean residual: {res_fou.mean():.6f}")
print(f"Fourier — L2 relative error: {l2_fou:.6f}")
np.savetxt("test_fourier.dat", np.hstack([X_ref, y_ref, y_pred_fou]))

plot_loss_and_l2(fourier_history, label=f"Fourier_sigma{CFG['sigma']}")
plot_solution(y_pred_fou,         label=f"Fourier_sigma{CFG['sigma']}")
plot_slices(y_pred_fou,           label=f"Fourier_sigma{CFG['sigma']}")


# Comparison
plot_comparison(vanilla_history, fourier_history,
                CFG["sigma"], l2_van, l2_fou)


# #── 12. Sigma sweep (uncomment once single run verified) ──────────────────────

# sigma_vals = [1, 2, 3, 4, 5, 7, 10]
# l2_errors  = []

# for sigma in sigma_vals:
#     print(f"\n--- sigma = {sigma} ---")
#     CFG["sigma"] = sigma
#     m   = FourierPINN().to(device)
#     h   = train(m, label=f"sigma{sigma}")
#     l2  = l2_relative_error(m)
#     l2_errors.append(l2)
#     print(f"  sigma={sigma}  L2={l2:.6f}")

# d_val = CFG["d"]
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.semilogy(sigma_vals, l2_errors, "o-", lw=2, ms=8)
# ax.set_xlabel("σ"); ax.set_ylabel("L2 relative error (log)")
# ax.set_title("Effect of Fourier feature scale")
# ax.legend(); ax.grid(True, which="both", alpha=0.3)
# plt.tight_layout(); plt.savefig("sigma_sweep.png", dpi=150); plt.show()