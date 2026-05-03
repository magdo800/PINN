""" Visualise SAPINN weight evolution, as discussed in dissertation section 4.2"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

def to_numpy(t):
    return t.detach().cpu().numpy()


# ---------------- PINN MODEL ----------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully connected network
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Xavier initialization for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# Problem parameters
eps = 2.5
beta = 1
x0 = 0.3


# ---------------- PROBLEM DEFINITION ----------------
def u_exact(x, eps=2.5, beta=1, x0=0.3):
    return torch.sin(6 * torch.pi * x) + eps * torch.exp(-beta * (x - x0)**2)

def f(x, eps=2.5, beta=1, x0=0.3):
    term1 = 36 * torch.pi**2 * torch.sin(6 * torch.pi * x)
    term2 = eps * torch.exp(-beta * (x - x0)**2) * (2*beta - 4*beta**2 * (x - x0)**2)
    return term1 + term2


# ---------------- PDE RESIDUAL ----------------
def residual(model, x):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)

    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return -u_xx - f(x)


# ---------------- DATA ----------------
torch.manual_seed(2)

N_r = 50

# Collocation points
x_r = torch.rand(N_r, 1).to(device)

# Boundary points
x_b = torch.tensor([[0.0], [1.0]]).to(device)

# Test grid
x_test = torch.linspace(0, 1, 500).view(-1, 1).to(device)
u_true_test = u_exact(x_test)


# ---------------- MODELS ----------------
model_pinn = PINN().to(device)
model_sa = PINN().to(device)
model_sa.load_state_dict(model_pinn.state_dict())

opt_pinn = torch.optim.Adam(model_pinn.parameters(), lr=1e-3)
opt_sa = torch.optim.Adam(model_sa.parameters(), lr=1e-3)

# Adaptive SA weights (learnable parameters)
lambda_r = nn.Parameter(torch.ones(N_r, 1, device=device))
lambda_b = nn.Parameter(torch.ones(2, 1, device=device))

# Optimiser for weight updates (gradient ascent)
opt_lambda = torch.optim.Adam([lambda_r, lambda_b], lr=1e-1, maximize=True)


# ---------------- SNAPSHOT STORAGE ----------------
snapshot_every = 200
snapshots = []  # stores (epoch, collocation points, weights)


# ---------------- TRAINING ----------------
epochs = 20000
resample_every = 20000

error_pinn, error_sa = [], []

for epoch in range(epochs):

    # ----- periodic resampling of collocation points -----
    if epoch % resample_every == 0 and epoch > 0:
        x_r = torch.rand(N_r, 1).to(device)

        # reset weights
        with torch.no_grad():
            lambda_r.data = torch.ones(N_r, 1, device=device)
            lambda_b.data = torch.ones(2, 1, device=device)

        opt_lambda = torch.optim.Adam([lambda_r, lambda_b], lr=1e-2, maximize=True)


    # ----- Vanilla PINN update -----
    opt_pinn.zero_grad()

    R = residual(model_pinn, x_r)
    L_r = torch.mean(R**2)

    B = model_pinn(x_b) - u_exact(x_b)
    L_b = torch.mean(B**2)

    loss_pinn = L_r + L_b
    loss_pinn.backward()
    opt_pinn.step()


    # ----- SA-PINN update -----
    opt_sa.zero_grad()
    opt_lambda.zero_grad()

    R_sa = residual(model_sa, x_r)
    L_r_sa = torch.mean(lambda_r * R_sa**2)

    B_sa = model_sa(x_b) - u_exact(x_b)
    L_b_sa = torch.mean(lambda_b * B_sa**2)

    loss_sa = L_r_sa + L_b_sa
    loss_sa.backward()

    opt_sa.step()
    opt_lambda.step()

    # ----- enforce positivity + normalisation of weights -----
    with torch.no_grad():
        lambda_r.clamp_(min=0)
        lambda_b.clamp_(min=0)

        lambda_r /= (lambda_r.mean() + 1e-8)
        lambda_b /= (lambda_b.mean() + 1e-8)


    # ----- store snapshots for animation -----
    if epoch % snapshot_every == 0:
        snapshots.append((
            epoch,
            x_r.cpu().numpy().copy(),
            lambda_r.detach().cpu().numpy().copy()
        ))


    # ----- error tracking -----
    with torch.no_grad():
        u_p = model_pinn(x_test)
        u_s = model_sa(x_test)

        err_p = torch.norm(u_p - u_true_test) / torch.norm(u_true_test)
        err_s = torch.norm(u_s - u_true_test) / torch.norm(u_true_test)

        error_pinn.append(err_p.item())
        error_sa.append(err_s.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | PINN: {err_p:.2e} | SA: {err_s:.2e}")


# ---------------- ANIMATION ----------------
fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                         gridspec_kw={'height_ratios': [1, 2]})

ax_w = axes[0]  # weights
ax_u = axes[1]  # solution

x_np = to_numpy(x_test)
u_true_np = to_numpy(u_true_test)

# true solution
ax_u.plot(x_np, u_true_np, 'k-', lw=1.5, label="True")
line_sa_u, = ax_u.plot([], [], 'r--', lw=1.5, label="SA-PINN")

ax_u.set_xlim(0, 1)
ax_u.set_ylim(u_true_np.min() - 0.5, u_true_np.max() + 0.5)
ax_u.set_xlabel("x")
ax_u.set_ylabel("u(x)")
ax_u.legend()

# weight scatter plot
scat = ax_w.scatter([], [], c=[], cmap="plasma", s=40, vmin=0, vmax=3)
ax_w.set_xlim(0, 1)
ax_w.set_ylim(0, 4)
ax_w.set_xlabel("x")
ax_w.set_ylabel("λ")
plt.colorbar(scat, ax=ax_w, label="λ value")

title = fig.suptitle("")
plt.tight_layout()


# animation update function
def update(frame):
    epoch, x_snap, lam_snap = snapshots[frame]

    x_flat = x_snap.flatten()
    lam_flat = lam_snap.flatten()

    scat.set_offsets(np.column_stack([x_flat, lam_flat]))
    scat.set_array(lam_flat)

    x_t = torch.linspace(0, 1, 500).view(-1, 1).to(device)

    with torch.no_grad():
        u_s = model_sa(x_t).cpu().numpy()

    line_sa_u.set_data(x_np, u_s)

    title.set_text(f"Epoch {epoch}")
    return scat, line_sa_u, title


ani = animation.FuncAnimation(
    fig, update, frames=len(snapshots), interval=80, blit=True
)

ani.save("sa_weights_evolution.gif", writer="pillow", fps=15)
print("Saved sa_weights_evolution.gif")

plt.show()
