# -*- coding: utf-8 -*-
import os, time, math, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import trange

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
print("Device:", DEVICE)

OUTDIR = "burgers_asw_outputs"
os.makedirs(OUTDIR, exist_ok=True)

NU = 0.01 / math.pi

WIDTH = 160
N_INT = 96
N_BND = 36
STEPS = 1500
LR = 1.5e-3
SEEDS = [0, 1, 2]

BETA_GRID_N = 41
WEIGHT_UPDATE_EVERY = 25
LOG_EVERY = 10
L2_GRID = 101

RIDGE = 1e-10

EMA_ALPHA_LOSS = 0.06
EMA_ALPHA_L2 = 0.08
EMA_ALPHA_COMPONENT = 0.06
EMA_ALPHA_BETA = 0.12

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Manufactured forced Burgers solution
# u_t + u u_x - nu u_xx = f
# exact u = exp(-t) sin(pi x)
# Domain: x in [-1, 1], t in [0, 1]
def u_exact_xt(x, t):
    return torch.exp(-t) * torch.sin(math.pi * x)

def forcing_xt(x, t):
    u = u_exact_xt(x, t)
    u_t = -torch.exp(-t) * torch.sin(math.pi * x)
    u_x = math.pi * torch.exp(-t) * torch.cos(math.pi * x)
    u_xx = -(math.pi ** 2) * torch.exp(-t) * torch.sin(math.pi * x)
    return u_t + u * u_x - NU * u_xx

# Two-layer ReLU^3 network
class TwoLayerReLU3(nn.Module):
    def __init__(self, width=160):
        super().__init__()
        self.W = nn.Parameter(torch.randn(width, 2, dtype=DTYPE) / math.sqrt(2))
        self.b = nn.Parameter(torch.randn(width, dtype=DTYPE) * 0.1)

        a = torch.randint(0, 2, (width,), dtype=DTYPE) * 2 - 1
        self.register_buffer("a", a)

    def forward(self, X):
        z = X @ self.W.T + self.b
        h = torch.relu(z) ** 3
        return (h @ self.a.view(-1, 1)) / math.sqrt(self.W.shape[0])

# Fixed collocation points
def sample_points(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 12345)

    # interior
    x = -1 + 2 * torch.rand(N_INT, 1, generator=g)
    t = torch.rand(N_INT, 1, generator=g)
    X_int = torch.cat([x, t], dim=1)

    # boundary / initial: split among t=0, x=-1, x=1
    n_each = N_BND // 3

    x0 = -1 + 2 * torch.rand(n_each, 1, generator=g)
    t0 = torch.zeros(n_each, 1)
    init = torch.cat([x0, t0], dim=1)

    tl = torch.rand(n_each, 1, generator=g)
    left = torch.cat([-torch.ones(n_each, 1), tl], dim=1)

    tr = torch.rand(n_each, 1, generator=g)
    right = torch.cat([torch.ones(n_each, 1), tr], dim=1)

    X_bnd = torch.cat([init, left, right], dim=0)

    return X_int.to(DEVICE), X_bnd.to(DEVICE)

# Residuals
def interior_residual_vector(model, X_int):
    X = X_int.detach().clone().requires_grad_(True)
    x = X[:, 0:1]
    t = X[:, 1:2]

    u = model(X)

    grad_u = torch.autograd.grad(
        u,
        X,
        torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_ux = torch.autograd.grad(
        u_x,
        X,
        torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]

    u_xx = grad_ux[:, 0:1]

    f = forcing_xt(x, t)
    r = u_t + u * u_x - NU * u_xx - f

    return r.reshape(-1) / math.sqrt(N_INT)

def boundary_residual_vector(model, X_bnd):
    x = X_bnd[:, 0:1]
    t = X_bnd[:, 1:2]

    u = model(X_bnd)
    u_true = u_exact_xt(x, t)

    return (u - u_true).reshape(-1) / math.sqrt(N_BND)

def component_losses(model, X_int, X_bnd):
    r_int = interior_residual_vector(model, X_int)
    r_bnd = boundary_residual_vector(model, X_bnd)

    li = 0.5 * torch.sum(r_int ** 2)
    lb = 0.5 * torch.sum(r_bnd ** 2)

    return li, lb

# Flatten parameters
def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])

# Jacobian rows wrt parameters
def jacobian_residuals(model, residual_vec):
    params = trainable_params(model)
    rows = []

    for i in range(residual_vec.numel()):
        grads = torch.autograd.grad(
            residual_vec[i],
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False
        )
        rows.append(flatten_grads(grads))

    return torch.stack(rows, dim=0)

@torch.no_grad()
def smallest_eigval_psd(M):
    M = 0.5 * (M + M.T)
    vals = torch.linalg.eigvalsh(M)
    return vals[0].item()

# True ASW beta selection
# For J=2:
# D1 = interior Jacobian, D2 = boundary Jacobian
# Hji = Dj Di^T
# J_1 = stacked [H11; H21], J_2 = stacked [H12; H22]
# G(beta)= beta J1 J1^T + (1-beta) J2 J2^T
# choose beta maximizing lambda_min(G)
def select_beta_asw(model, X_int, X_bnd):
    model.zero_grad(set_to_none=True)

    r1 = interior_residual_vector(model, X_int)
    r2 = boundary_residual_vector(model, X_bnd)

    D1 = jacobian_residuals(model, r1)
    D2 = jacobian_residuals(model, r2)

    H11 = D1 @ D1.T
    H12 = D1 @ D2.T
    H21 = D2 @ D1.T
    H22 = D2 @ D2.T

    J1 = torch.cat([H11, H21], dim=0)
    J2 = torch.cat([H12, H22], dim=0)

    A1 = J1 @ J1.T
    A2 = J2 @ J2.T

    n = A1.shape[0]
    I = torch.eye(n, device=DEVICE, dtype=DTYPE)

    betas = torch.linspace(0.0, 1.0, BETA_GRID_N, device=DEVICE)

    best_beta = 0.5
    best_val = -float("inf")

    for b in betas:
        G = b * A1 + (1.0 - b) * A2 + RIDGE * I
        val = smallest_eigval_psd(G)

        if val > best_val:
            best_val = val
            best_beta = float(b.item())

    return best_beta, best_val

# Relative L2 error
@torch.no_grad()
def relative_l2(model):
    xs = torch.linspace(-1, 1, L2_GRID, device=DEVICE).view(-1, 1)
    ts = torch.linspace(0, 1, L2_GRID, device=DEVICE).view(-1, 1)

    Xg, Tg = torch.meshgrid(xs.squeeze(), ts.squeeze(), indexing="ij")
    XT = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)

    pred = model(XT).reshape(-1)
    true = u_exact_xt(XT[:, 0:1], XT[:, 1:2]).reshape(-1)

    return (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()

def ema(y, alpha):
    y = np.asarray(y, dtype=float)

    if len(y) == 0:
        return y

    out = np.zeros_like(y)
    out[0] = y[0]

    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]

    return out

# Train one run
def run_one(seed, mode="EW"):
    set_seed(seed)

    X_int, X_bnd = sample_points(seed)

    model = TwoLayerReLU3(WIDTH).to(DEVICE)
    opt = torch.optim.SGD(trainable_params(model), lr=LR)

    hist = {
        "loss": [],
        "rel_l2": [],
        "interior": [],
        "boundary": [],
        "beta": [],
        "eig": [],
        "steps": [],
    }

    beta = 0.5
    eig_val = np.nan

    start = time.time()
    pbar = trange(STEPS + 1, desc=f"{mode} seed={seed}", leave=False)

    for k in pbar:
        if mode == "ASW" and k % WEIGHT_UPDATE_EVERY == 0:
            beta, eig_val = select_beta_asw(model, X_int, X_bnd)

        li, lb = component_losses(model, X_int, X_bnd)
        loss = beta * li + (1.0 - beta) * lb

        if k % LOG_EVERY == 0 or k == STEPS:
            l2 = relative_l2(model)

            hist["loss"].append(loss.item())
            hist["rel_l2"].append(l2)
            hist["interior"].append(li.item())
            hist["boundary"].append(lb.item())
            hist["beta"].append(beta)
            hist["eig"].append(eig_val)
            hist["steps"].append(k)

            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                l2=f"{l2:.3f}",
                beta=f"{beta:.3f}"
            )

        if k == STEPS:
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    hist["runtime_sec"] = time.time() - start
    return hist

# Run paired experiment
all_results = {"EW": [], "ASW": []}

for seed in SEEDS:
    print(f"\n=== Seed {seed}: EW ===")
    all_results["EW"].append(run_one(seed, "EW"))

    print(f"\n=== Seed {seed}: ASW ===")
    all_results["ASW"].append(run_one(seed, "ASW"))

# Aggregate
def stack_metric(mode, key):
    return np.stack([np.asarray(r[key]) for r in all_results[mode]], axis=0)

steps = np.asarray(all_results["EW"][0]["steps"])

summary = {}

for mode in ["EW", "ASW"]:
    summary[mode] = {}

    for key in ["loss", "rel_l2", "interior", "boundary", "beta"]:
        arr = stack_metric(mode, key)
        summary[mode][key + "_mean"] = arr.mean(axis=0)
        summary[mode][key + "_std"] = arr.std(axis=0)

# Final table
ew_l2 = stack_metric("EW", "rel_l2")[:, -1]
asw_l2 = stack_metric("ASW", "rel_l2")[:, -1]

ew_loss = stack_metric("EW", "loss")[:, -1]
asw_loss = stack_metric("ASW", "loss")[:, -1]

ew_bnd = stack_metric("EW", "boundary")[:, -1]
asw_bnd = stack_metric("ASW", "boundary")[:, -1]

ew_int = stack_metric("EW", "interior")[:, -1]
asw_int = stack_metric("ASW", "interior")[:, -1]

print("\nFINAL RESULTS")
print(f"Final relative L2: EW  {ew_l2.mean():.4f} ± {ew_l2.std():.4f}")
print(f"Final relative L2: ASW {asw_l2.mean():.4f} ± {asw_l2.std():.4f}")
print(f"Relative change: {(asw_l2.mean() / ew_l2.mean() - 1) * 100:.1f}%")
print(f"Win rate ASW: {(asw_l2 < ew_l2).sum()}/{len(SEEDS)}")
print()
print(f"Final weighted loss: EW  {ew_loss.mean():.4e} ± {ew_loss.std():.4e}")
print(f"Final weighted loss: ASW {asw_loss.mean():.4e} ± {asw_loss.std():.4e}")
print()
print(f"Final interior loss: EW  {ew_int.mean():.4e} ± {ew_int.std():.4e}")
print(f"Final interior loss: ASW {asw_int.mean():.4e} ± {asw_int.std():.4e}")
print()
print(f"Final boundary loss: EW  {ew_bnd.mean():.4e} ± {ew_bnd.std():.4e}")
print(f"Final boundary loss: ASW {asw_bnd.mean():.4e} ± {asw_bnd.std():.4e}")

# Save raw data
np.savez(
    f"{OUTDIR}/burgers_results_raw.npz",
    steps=steps,
    ew_loss=stack_metric("EW", "loss"),
    asw_loss=stack_metric("ASW", "loss"),
    ew_rel_l2=stack_metric("EW", "rel_l2"),
    asw_rel_l2=stack_metric("ASW", "rel_l2"),
    ew_interior=stack_metric("EW", "interior"),
    asw_interior=stack_metric("ASW", "interior"),
    ew_boundary=stack_metric("EW", "boundary"),
    asw_boundary=stack_metric("ASW", "boundary"),
    ew_beta=stack_metric("EW", "beta"),
    asw_beta=stack_metric("ASW", "beta"),
)

# Plot settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
})

# Plot 1: training loss and relative L2
plt.figure(figsize=(11, 4.2))

plt.subplot(1, 2, 1)
plt.plot(
    steps,
    ema(summary["EW"]["loss_mean"], EMA_ALPHA_LOSS),
    label="EW"
)
plt.plot(
    steps,
    ema(summary["ASW"]["loss_mean"], EMA_ALPHA_LOSS),
    label="ASW"
)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Weighted training loss")
plt.title("Burgers: Training loss")
plt.legend()
plt.grid(alpha=0.25)

plt.subplot(1, 2, 2)
plt.plot(
    steps,
    ema(summary["EW"]["rel_l2_mean"], EMA_ALPHA_L2),
    label="EW"
)
plt.plot(
    steps,
    ema(summary["ASW"]["rel_l2_mean"], EMA_ALPHA_L2),
    label="ASW"
)
plt.xlabel("Iteration")
plt.ylabel("Relative $L^2$ error")
plt.title("Burgers: Relative $L^2$ error")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_burgers_loss_l2.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2: interior and boundary losses
plt.figure(figsize=(11, 4.2))

plt.subplot(1, 2, 1)
plt.plot(
    steps,
    ema(summary["EW"]["interior_mean"], EMA_ALPHA_COMPONENT),
    label="EW"
)
plt.plot(
    steps,
    ema(summary["ASW"]["interior_mean"], EMA_ALPHA_COMPONENT),
    label="ASW"
)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Interior loss")
plt.title("Burgers: Interior residual")
plt.legend()
plt.grid(alpha=0.25)

plt.subplot(1, 2, 2)
plt.plot(
    steps,
    ema(summary["EW"]["boundary_mean"], EMA_ALPHA_COMPONENT),
    label="EW"
)
plt.plot(
    steps,
    ema(summary["ASW"]["boundary_mean"], EMA_ALPHA_COMPONENT),
    label="ASW"
)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Boundary / initial loss")
plt.title("Burgers: Boundary loss")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_burgers_components.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3: beta trajectory
plt.figure(figsize=(6.5, 4.2))

plt.plot(
    steps,
    summary["EW"]["beta_mean"],
    label="EW fixed $\\beta=0.5$"
)
plt.plot(
    steps,
    ema(summary["ASW"]["beta_mean"], EMA_ALPHA_BETA),
    label="ASW $\\beta(k)$"
)

plt.xlabel("Iteration")
plt.ylabel("$\\beta$ on interior residual")
plt.title("Burgers: Adaptive weight trajectory")
plt.ylim(-0.02, 1.02)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig3_burgers_beta.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figures and raw results to: {OUTDIR}/")
