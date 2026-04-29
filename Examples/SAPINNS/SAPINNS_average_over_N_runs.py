" Training sweep for SAPINN 50 collocation points, averaged over N runs, as discussed in dissertation section 4.2"
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- PINN MODEL ----------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully connected MLP shape [1,64,64,1]
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- PROBLEM DEFINITION ----------------
def u_exact(x, eps=2.5, beta=10, x0=0.3):
    return torch.sin(6 * torch.pi * x) + eps * torch.exp(-beta * (x - x0)**2)

def f(x, eps=2.5, beta=10, x0=0.3):
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


# ---------------- SINGLE EXPERIMENT RUN ----------------
def run_single(seed, epochs=20000, N_r=50, lr_lambda=1e-2):
    torch.manual_seed(seed)

    # Training points
    x_r = torch.rand(N_r, 1).to(device)
    x_b = torch.tensor([[0.0], [1.0]]).to(device)

    # Test grid
    x_test = torch.linspace(0, 1, 500).view(-1, 1).to(device)
    u_true_test = u_exact(x_test)

    # Models
    model_pinn = PINN().to(device)
    model_sa = PINN().to(device)
    model_sa.load_state_dict(model_pinn.state_dict())

    # Optimizers
    opt_pinn = torch.optim.Adam(model_pinn.parameters(), lr=1e-3)
    opt_sa = torch.optim.Adam(model_sa.parameters(), lr=1e-3)

    # SA adaptive weights
    lambda_r = torch.ones(N_r, 1, device=device)
    lambda_b = torch.ones(2, 1, device=device)

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(epochs):

        # ----- Vanilla PINN -----
        opt_pinn.zero_grad()

        R = residual(model_pinn, x_r)
        L_r = torch.mean(R**2)

        B = model_pinn(x_b) - u_exact(x_b)
        L_b = torch.mean(B**2)

        (L_r + L_b).backward()
        opt_pinn.step()

        # ----- SA-PINN -----
        opt_sa.zero_grad()

        R_sa = residual(model_sa, x_r)
        L_r_sa = torch.mean(lambda_r * R_sa**2)

        B_sa = model_sa(x_b) - u_exact(x_b)
        L_b_sa = torch.mean(lambda_b * B_sa**2)

        (L_r_sa + L_b_sa).backward()
        opt_sa.step()

        # Update SA weights
        with torch.no_grad():
            lambda_r += lr_lambda * (R_sa**2)
            lambda_b += lr_lambda * (B_sa**2)

            # Normalize weights (stability)
            lambda_r /= (lambda_r.mean() + 1e-8)
            lambda_b /= (lambda_b.mean() + 1e-8)

    # ---------------- EVALUATION ----------------
    with torch.no_grad():
        err_pinn = (torch.norm(model_pinn(x_test) - u_true_test) /
                    torch.norm(u_true_test)).item()

        err_sa = (torch.norm(model_sa(x_test) - u_true_test) /
                  torch.norm(u_true_test)).item()

    return err_pinn, err_sa


# ---------------- EXPERIMENT SETUP ----------------
seeds = list(range(5))

def geo_mean(x): return np.exp(np.mean(np.log(x)))
def geo_std(x):  return np.exp(np.std(np.log(x)))


# ---------------- RUN EXPERIMENTS ----------------
results_pinn = []
results_sa = []

print(f"{'Seed':<6} {'PINN':>12} {'SA':>12}")
print("-" * 32)

for seed in seeds:
    err_pinn, err_sa = run_single(seed)

    results_pinn.append(err_pinn)
    results_sa.append(err_sa)

    print(f"{seed:<6} {err_pinn:>12.3e} {err_sa:>12.3e}")

results_pinn = np.array(results_pinn)
results_sa = np.array(results_sa)


# ---------------- SUMMARY STATISTICS ----------------
col_w = 14

print(f"\n{'='*58}")
print(f"{'':16} {'Vanilla PINN':>{col_w}} {'SA-PINN':>{col_w}}")
print(f"{'='*58}")

print(f"{'Arith mean':<16} {np.mean(results_pinn):>{col_w}.3e} {np.mean(results_sa):>{col_w}.3e}")
print(f"{'Arith std':<16} {np.std(results_pinn):>{col_w}.3e} {np.std(results_sa):>{col_w}.3e}")

print(f"{'Geo mean':<16} {geo_mean(results_pinn):>{col_w}.3e} {geo_mean(results_sa):>{col_w}.3e}")
print(f"{'Geo std':<16} {geo_std(results_pinn):>{col_w}.3f} {geo_std(results_sa):>{col_w}.3f}")

print(f"{'='*58}")

print(f"{'Geo mean ratio (SA/PINN)':<28} "
      f"{geo_mean(results_sa)/geo_mean(results_pinn):.3f}")

print(f"{'='*58}")
