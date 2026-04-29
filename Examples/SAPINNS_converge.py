""

import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import time

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

def to_numpy(t):
    return t.detach().cpu().numpy()

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


# Problem parameters
eps = 2.5
beta = 10
x0 = 0.3

# ---------------- EXACT SOLUTION / FORCING ----------------
def u_exact(x, eps=2.5, beta=10, x0=0.3):
    return torch.sin(6 * torch.pi * x) + eps * torch.exp(-beta * (x - x0)**2)

def f(x, eps=2.5, beta=10, x0=0.3):
    term1 = 36 * torch.pi**2 * torch.sin(6 * torch.pi * x)
    term2 = eps * torch.exp(-beta * (x - x0)**2) * (2*beta - 4*beta**2 * (x - x0)**2)
    return term1 + term2

# PDE residual
def residual(model, x):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)

    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return -u_xx - f(x)


# ---------------- DATA ----------------
torch.manual_seed(0)

N_r = 50  # collocation points
x_r = torch.rand(N_r, 1).to(device)

# boundary points
x_b = torch.tensor([[0.0], [1.0]]).to(device)

# test grid
x_test = torch.linspace(0, 1, 500).view(-1, 1).to(device)
u_true_test = u_exact(x_test)


# ---------------- MODELS ----------------
model_pinn = PINN().to(device)
model_sa   = PINN().to(device)

# initialize SA model with same weights
model_sa.load_state_dict(model_pinn.state_dict())

opt_pinn = torch.optim.Adam(model_pinn.parameters(), lr=1e-3)
opt_sa   = torch.optim.Adam(model_sa.parameters(), lr=1e-3)

# SA weighting parameters
lambda_r = torch.ones(N_r, 1, device=device)
lambda_b = torch.ones(2, 1, device=device)
lr_lambda = 1e-2


# ---------------- TRAINING ----------------
epochs = 20000
error_pinn, error_sa = [], []

for epoch in range(epochs):

    # ----- Vanilla PINN -----
    opt_pinn.zero_grad()

    R = residual(model_pinn, x_r)
    L_r = torch.mean(R**2)

    B = model_pinn(x_b) - u_exact(x_b)
    L_b = torch.mean(B**2)

    loss_pinn = L_r + L_b
    loss_pinn.backward()
    opt_pinn.step()



    # ----- SA-PINN -----
    opt_sa.zero_grad()

    R_sa = residual(model_sa, x_r)
    L_r_sa = torch.mean(lambda_r * R_sa**2)

    B_sa = model_sa(x_b) - u_exact(x_b)
    L_b_sa = torch.mean(lambda_b * B_sa**2)

    loss_sa = L_r_sa + L_b_sa
    loss_sa.backward()
    opt_sa.step()

    # update SA weights
    with torch.no_grad():
        lambda_r += lr_lambda * (R_sa**2)
        lambda_b += lr_lambda * (B_sa**2)

        lambda_r /= (lambda_r.mean() + 1e-8)
        lambda_b /= (lambda_b.mean() + 1e-8)



    # ----- evaluation -----
    with torch.no_grad():
        u_p = model_pinn(x_test)
        u_s = model_sa(x_test)

        err_p = torch.norm(u_p - u_true_test) / torch.norm(u_true_test)
        err_s = torch.norm(u_s - u_true_test) / torch.norm(u_true_test)

        error_pinn.append(err_p.item())
        error_sa.append(err_s.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | PINN: {err_p:.2e} | SA: {err_s:.2e}")


# ---------------- RESULTS ----------------

# error curve
plt.figure()
plt.plot(error_pinn, label="Vanilla PINN")
plt.plot(error_sa, label="SA-PINN")
plt.yscale("log")
plt.legend()
plt.title("Relative Error")
plt.xlabel("Epoch")
plt.ylabel("L2 Error")
plt.show()

# solution comparison
with torch.no_grad():
    u_p = model_pinn(x_test)
    u_s = model_sa(x_test)

plt.figure()
plt.plot(to_numpy(x_test), to_numpy(u_true_test), label="True")
plt.plot(to_numpy(x_test), to_numpy(u_p), '--', label="PINN")
plt.plot(to_numpy(x_test), to_numpy(u_s), ':', label="SA-PINN")
plt.legend()
plt.title("Solution Comparison")
plt.show()

# pointwise error
plt.figure()
plt.plot(to_numpy(x_test), to_numpy(torch.abs(u_p - u_true_test)), label="PINN")
plt.plot(to_numpy(x_test), to_numpy(torch.abs(u_s - u_true_test)), label="SA-PINN")
plt.legend()
plt.title("Pointwise Error")
plt.show()

# residual (SA model)
x_vis = torch.linspace(0, 1, 500).view(-1, 1).to(device)
R_vis = residual(model_sa, x_vis)

plt.figure()
plt.plot(to_numpy(x_vis), to_numpy(torch.abs(R_vis)))
plt.title("Residual (SA-PINN)")
plt.show()

# SA weights visualization
plt.figure()
plt.scatter(to_numpy(x_r), to_numpy(lambda_r), s=10)
plt.title("SA Weights")
plt.show()