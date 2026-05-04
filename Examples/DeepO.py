"""
PI-DeepONet: Advection Equation Experiment
==========================================
Learns the solution operator G: v -> u for the advection PDE:
    du/dt + du/dx = 0,  (x,t) in [0,1]^2
    u(x, 0) = v(x)
 
where v is drawn from a periodic GRF (ExpSineSquared kernel).
Exact solution: u(x, t) = v(x - t)
 
Figures produced:
    1. GRF samples + exact solution heatmaps  (fig1_grf_samples.png)
    2. Training loss curve                    (fig2_loss.png)
    3. Pred vs True vs Error for one test fn  (fig3_pred_vs_true.png)
    4. L2 error distribution over test set    (fig4_error_dist.png)
    5. Error vs number of sensors             (fig5_sensor_ablation.png)
"""
 
# ── Backend ───────────────────────────────────────────────────────────────────
import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
 
if dde.backend.backend_name == "paddle":
    import paddle
    dim_x = 5
    sin = paddle.sin
    cos = paddle.cos
    concat = paddle.concat
elif dde.backend.backend_name == "pytorch":
    import torch
    dim_x = 5
    sin = torch.sin
    cos = torch.cos
    concat = torch.cat
else:
    from deepxde.backend import tf
    dim_x = 2
    sin = tf.sin
    cos = tf.cos
    concat = tf.concat
 
os.makedirs("figures", exist_ok=True)
 
# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "text.usetex": False,
})
 
# ── Helper: build and train one model ────────────────────────────────────────
def build_and_train(num_sensors=50, iterations=30000, seed=0):
    """Build a PI-DeepONet for the advection equation and train it."""
    dde.config.set_random_seed(seed)
 
    # PDE residual: du/dt + du/dx = 0
    def pde(x, y, v):
        dy_x = dde.grad.jacobian(y, x, j=0)
        dy_t = dde.grad.jacobian(y, x, j=1)
        return dy_t + dy_x
 
    # Geometry: treat (x, t) as two spatial coords on [0,1]^2
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
 
    # Initial condition u(x, 0) = v(x)
    def func_ic(x, v):
        return v
 
    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[1], 0)
 
    ic = dde.icbc.DirichletBC(geom, func_ic, boundary)
    pde_data = dde.data.PDE(geom, pde, ic, num_domain=200, num_boundary=200)
 
    # Function space: periodic GRF
    func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
 
    # Sensor locations (branch net inputs)
    eval_pts = np.linspace(0, 1, num=num_sensors)[:, None]
 
    data = dde.data.PDEOperatorCartesianProd(
        pde_data, func_space, eval_pts, 1000,
        function_variables=[0], num_test=100, batch_size=32
    )
 
    # Network: branch [num_sensors -> 128x3], trunk [dim_x -> 128x3]
    net = dde.nn.DeepONetCartesianProd(
        [num_sensors, 128, 128, 128],
        [dim_x, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )
 
    # Periodic feature transform on the trunk (x coordinate only)
    def periodic(x):
        x_coord, t = x[:, :1], x[:, 1:]
        x_coord = x_coord * 2 * np.pi
        return concat([cos(x_coord), sin(x_coord),
                       cos(2 * x_coord), sin(2 * x_coord), t], 1)
 
    net.apply_feature_transform(periodic)
 
    model = dde.Model(data, net)
    model.compile("adam", lr=0.0005)
    losshistory, train_state = model.train(iterations=iterations)
 
    return model, losshistory, train_state, eval_pts, func_space
 
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: GRF input samples + exact solution heatmaps
# ═════════════════════════════════════════════════════════════════════════════
def plot_grf_samples():
    print("Plotting Figure 1: GRF samples + exact solutions...")
    func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
    x_line = np.linspace(0, 1, 200)
    t_line = np.linspace(0, 1, 200)
 
    np.random.seed(42)
    n_samples = 4
    samples = func_space.random(n_samples)          # (n_samples, n_pts) -- not used directly
    # Evaluate v on x_line
    v_vals = func_space.eval_batch(samples, x_line[:, None])  # (n_samples, 200)
 
    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(2, n_samples, hspace=0.45, wspace=0.35)
 
    for i in range(n_samples):
        v = v_vals[i]   # shape (200,)
 
        # Top row: v(x)
        ax_top = fig.add_subplot(gs[0, i])
        ax_top.plot(x_line, v, color="#1f4e79", lw=1.8)
        ax_top.set_xlim(0, 1)
        ax_top.set_xlabel("$x$")
        if i == 0:
            ax_top.set_ylabel("$v(x)$")
        ax_top.set_title(f"Sample {i+1}")
        ax_top.tick_params(labelsize=9)
 
        # Bottom row: exact solution u(x,t) = v(x - t) as heatmap
        ax_bot = fig.add_subplot(gs[1, i])
        xv, tv = np.meshgrid(x_line, t_line)
        # v is defined on x_line; interpolate v(x - t)
        u_exact = np.array([
            np.interp((x_line - ti) % 1.0, x_line, v)
            for ti in t_line
        ])
        im = ax_bot.imshow(
            u_exact, extent=[0, 1, 0, 1], origin="lower",
            aspect="auto", cmap="RdBu_r"
        )
        ax_bot.set_xlabel("$x$")
        if i == 0:
            ax_bot.set_ylabel("$t$")
        ax_bot.tick_params(labelsize=9)
        plt.colorbar(im, ax=ax_bot, fraction=0.046, pad=0.04)
 
    fig.text(0.5, 1.01, "Input functions $v(x)$ (top) and exact solutions $u(x,t)=v(x-t)$ (bottom)",
             ha="center", fontsize=13)
    plt.savefig("figures/fig1_grf_samples.png", bbox_inches="tight")
    plt.close()
    print("  Saved figures/fig1_grf_samples.png")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING RUN
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Training main PI-DeepONet model (50 sensors, 30000 iters)")
print("="*60)
model, losshistory, train_state, eval_pts, func_space = build_and_train(
    num_sensors=50, iterations=30000, seed=0
)
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 (can be done before training)
# ═════════════════════════════════════════════════════════════════════════════
plot_grf_samples()
 
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Training loss curve
# ═════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 2: Loss curve...")
fig, ax = plt.subplots(figsize=(6, 3.5))
loss_train = np.sum(losshistory.loss_train, axis=1)
loss_test  = np.sum(losshistory.loss_test,  axis=1)
steps = losshistory.steps
 
ax.semilogy(steps, loss_train, label="Train loss", color="#1f4e79", lw=1.8)
ax.semilogy(steps, loss_test,  label="Test loss",  color="#c0392b", lw=1.8, ls="--")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss (log scale)")
ax.set_title("PI-DeepONet Training Loss — Advection Equation")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig2_loss.png", bbox_inches="tight")
plt.close()
print("  Saved figures/fig2_loss.png")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# Shared grid for prediction figures
# ═════════════════════════════════════════════════════════════════════════════
x_line = np.linspace(0, 1, 100)
t_line = np.linspace(0, 1, 100)
xv, tv = np.meshgrid(x_line, t_line)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T   # (10000, 2)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Predicted vs True vs Absolute Error (one test function)
# ═════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 3: Pred vs True vs Error...")
 
np.random.seed(7)
test_sample = func_space.random(1)
v_branch = func_space.eval_batch(test_sample, eval_pts)   # (1, 50)
 
# Exact solution: u(x,t) = v(x-t), using the full x_line for v
v_full = func_space.eval_batch(test_sample, x_line[:, None])[0]  # (100,)
u_true = np.array([
    np.interp((x_line - ti) % 1.0, x_line, v_full)
    for ti in t_line
])  # (100, 100)
 
u_pred = model.predict((v_branch, x_trunk)).reshape(100, 100)
abs_err = np.abs(u_pred - u_true)
 
vmin = min(u_true.min(), u_pred.min())
vmax = max(u_true.max(), u_pred.max())
 
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
titles = ["True $u(x,t)$", "Predicted $\\hat{u}(x,t)$", "Absolute error"]
data_list = [u_true, u_pred, abs_err]
cmaps = ["RdBu_r", "RdBu_r", "hot_r"]
 
for ax, dat, title, cmap in zip(axes, data_list, titles, cmaps):
    kwargs = dict(extent=[0,1,0,1], origin="lower", aspect="auto", cmap=cmap)
    if cmap == "RdBu_r":
        kwargs.update(vmin=vmin, vmax=vmax)
    im = ax.imshow(dat, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
l2_err = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
fig.suptitle(f"PI-DeepONet prediction — $L^2$ relative error: {l2_err:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig3_pred_vs_true.png", bbox_inches="tight")
plt.close()
print(f"  Saved figures/fig3_pred_vs_true.png  (L2 err = {l2_err:.4f})")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Distribution of L2 errors over test set
# ═════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 4: Error distribution...")
 
N_test = 200
np.random.seed(99)
test_samples = func_space.random(N_test)
v_branches = func_space.eval_batch(test_samples, eval_pts)   # (N_test, 50)
 
l2_errors = []
for i in range(N_test):
    v_i = v_branches[i:i+1]
    v_full_i = func_space.eval_batch(test_samples[i:i+1], x_line[:, None])[0]
    u_true_i = np.array([
        np.interp((x_line - ti) % 1.0, x_line, v_full_i)
        for ti in t_line
    ])
    u_pred_i = model.predict((v_i, x_trunk)).reshape(100, 100)
    err = np.linalg.norm(u_pred_i - u_true_i) / np.linalg.norm(u_true_i)
    l2_errors.append(err)
 
l2_errors = np.array(l2_errors)
 
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(l2_errors, bins=30, color="#1f4e79", edgecolor="white", alpha=0.85)
ax.axvline(np.mean(l2_errors),  color="#c0392b", lw=2, ls="--", label=f"Mean  = {np.mean(l2_errors):.4f}")
ax.axvline(np.median(l2_errors),color="#e67e22", lw=2, ls=":",  label=f"Median = {np.median(l2_errors):.4f}")
ax.set_xlabel("Relative $L^2$ error")
ax.set_ylabel("Count")
ax.set_title(f"Error distribution over {N_test} test functions")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig4_error_dist.png", bbox_inches="tight")
plt.close()
print(f"  Saved figures/fig4_error_dist.png")
print(f"  Mean L2 = {np.mean(l2_errors):.4f}, Median = {np.median(l2_errors):.4f}, Max = {np.max(l2_errors):.4f}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Error vs number of sensors (ablation)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning sensor ablation (Figure 5) — this trains multiple models...")
 
sensor_counts = [10, 20, 30, 50, 75, 100]
mean_errors   = []
std_errors    = []
 
for n_s in sensor_counts:
    print(f"  Training with {n_s} sensors...")
    m, _, _, ep, fs = build_and_train(num_sensors=n_s, iterations=15000, seed=1)
 
    eval_pts_s = np.linspace(0, 1, num=n_s)[:, None]
    np.random.seed(99)
    ts = fs.random(50)
    vb = fs.eval_batch(ts, eval_pts_s)
 
    errs = []
    for i in range(50):
        v_i = vb[i:i+1]
        v_full_i = fs.eval_batch(ts[i:i+1], x_line[:, None])[0]
        u_true_i = np.array([
            np.interp((x_line - ti) % 1.0, x_line, v_full_i)
            for ti in t_line
        ])
        u_pred_i = m.predict((v_i, x_trunk)).reshape(100, 100)
        errs.append(np.linalg.norm(u_pred_i - u_true_i) / np.linalg.norm(u_true_i))
 
    mean_errors.append(np.mean(errs))
    std_errors.append(np.std(errs))
    print(f"    Mean L2 = {mean_errors[-1]:.4f}")
 
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.errorbar(sensor_counts, mean_errors, yerr=std_errors,
            fmt="-o", color="#1f4e79", capsize=4, lw=2, markersize=6, label="Mean ± std")
ax.set_xlabel("Number of sensors $m$")
ax.set_ylabel("Mean relative $L^2$ error")
ax.set_title("Effect of sensor count on prediction accuracy")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig5_sensor_ablation.png", bbox_inches="tight")
plt.close()
print("  Saved figures/fig5_sensor_ablation.png")
 
print("\n" + "="*60)
print("All figures saved to ./figures/")
print("="*60)