" Gradient ascent sweep for SAPINN learning rate, as discussed in dissertation section 4.2"

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

        # Xavier initialization for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

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


# ---------------- SINGLE TRAINING RUN ----------------
def run_single(seed, lr_lambda, epochs=20000, N_r=50):
    torch.manual_seed(seed)

    # Collocation and boundary points
    x_r = torch.rand(N_r, 1).to(device)
    x_b = torch.tensor([[0.0], [1.0]]).to(device)

    # Test grid
    x_test = torch.linspace(0, 1, 500).view(-1, 1).to(device)
    u_true_test = u_exact(x_test)

    # Model + optimizer
    model_sa = PINN().to(device)
    opt_sa = torch.optim.Adam(model_sa.parameters(), lr=1e-3)

    # SA weights
    lambda_r = torch.ones(N_r, 1, device=device)
    lambda_b = torch.ones(2, 1, device=device)

    # Training loop
    for epoch in range(epochs):
        opt_sa.zero_grad()

        # Residual loss
        R_sa = residual(model_sa, x_r)
        L_r_sa = torch.mean(lambda_r * R_sa**2)

        # Boundary loss
        B_sa = model_sa(x_b) - u_exact(x_b)
        L_b_sa = torch.mean(lambda_b * B_sa**2)

        # Backprop
        (L_r_sa + L_b_sa).backward()
        opt_sa.step()

        # Update adaptive weights
        with torch.no_grad():
            lambda_r += lr_lambda * (R_sa**2)
            lambda_b += lr_lambda * (B_sa**2)

            # Normalize weights (prevents explosion)
            lambda_r /= (lambda_r.mean() + 1e-8)
            lambda_b /= (lambda_b.mean() + 1e-8)

    # Relative L2 error
    with torch.no_grad():
        err = (torch.norm(model_sa(x_test) - u_true_test) /
               torch.norm(u_true_test)).item()

    return err


# ---------------- EXPERIMENT SETUP ----------------
lr_lambdas = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
seeds = [0, 1, 2]

# Log-space statistics
def geo_mean(x): return np.exp(np.mean(np.log(x)))
def geo_std(x):  return np.exp(np.std(np.log(x)))


print(f"Running {len(lr_lambdas)} lr_lambda values x {len(seeds)} seeds...\n")

all_results = {}

# ---------------- RUN GRID SEARCH ----------------
for lr in lr_lambdas:
    errors = []

    for seed in seeds:
        err = run_single(seed, lr_lambda=lr)
        errors.append(err)

        print(f"  lr_lambda={lr:.3f} | seed={seed} | err={err:.3e}")

    all_results[lr] = np.array(errors)


# ---------------- SUMMARY TABLE ----------------
col_w = 12

header = (f"{'lr_lambda':<12}"
          f"{'arith mean':>{col_w}}"
          f"{'arith std':>{col_w}}"
          f"{'geo mean':>{col_w}}"
          f"{'geo std':>{col_w}}")

print(f"\n{'='*60}")
print(header)
print(f"{'='*60}")

for lr, errors in all_results.items():
    print(f"{lr:<12.3f}"
          f"{np.mean(errors):>{col_w}.3e}"
          f"{np.std(errors):>{col_w}.3e}"
          f"{geo_mean(errors):>{col_w}.3e}"
          f"{geo_std(errors):>{col_w}.3f}")

print(f"{'='*60}")
