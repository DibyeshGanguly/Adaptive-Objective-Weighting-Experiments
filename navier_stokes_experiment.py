# -*- coding: utf-8 -*-
import os, time, math, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
print("Device:", DEVICE)

OUTDIR = "ns_asw_gradnorm_outputs"
os.makedirs(OUTDIR, exist_ok=True)

NU = 0.01

WIDTH = 160
N_INT = 96
N_BND = 40
STEPS = 1500
LR = 1.5e-3
SEEDS = [0, 1, 2]

MODES = ["EW", "GradNorm", "ASW"]

BETA_GRID_N = 41
WEIGHT_UPDATE_EVERY = 25
LOG_EVERY = 10
L2_GRID = 81

RIDGE = 1e-10
GRAD_EPS = 1e-12

EMA_ALPHA_LOSS = 0.025
EMA_ALPHA_L2 = 0.06
EMA_ALPHA_COMPONENT = 0.045
EMA_ALPHA_BETA = 0.06


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exact_uvp(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    u = torch.sin(math.pi * x) * torch.cos(math.pi * y)
    v = -torch.cos(math.pi * x) * torch.sin(math.pi * y)
    p = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return u, v, p


def forcing_ns(X):
    x = X[:, 0:1]
    y = X[:, 1:2]

    sx, cx = torch.sin(math.pi * x), torch.cos(math.pi * x)
    sy, cy = torch.sin(math.pi * y), torch.cos(math.pi * y)

    u = sx * cy
    v = -cx * sy

    u_x = math.pi * cx * cy
    u_y = -math.pi * sx * sy
    v_x = math.pi * sx * sy
    v_y = -math.pi * cx * cy

    p_x = math.pi * cx * sy
    p_y = math.pi * sx * cy

    lap_u = -2 * (math.pi ** 2) * u
    lap_v = -2 * (math.pi ** 2) * v

    f1 = u * u_x + v * u_y + p_x - NU * lap_u
    f2 = u * v_x + v * v_y + p_y - NU * lap_v
    return f1, f2


class TwoLayerReLU3NS(nn.Module):
    def __init__(self, width=160):
        super().__init__()
        self.W = nn.Parameter(torch.randn(width, 2, dtype=DTYPE) / math.sqrt(2))
        self.b = nn.Parameter(torch.randn(width, dtype=DTYPE) * 0.1)
        a = torch.randint(0, 2, (width, 3), dtype=DTYPE) * 2 - 1
        self.register_buffer("a", a)

    def forward(self, X):
        z = X @ self.W.T + self.b
        h = torch.relu(z) ** 3
        return h @ self.a / math.sqrt(self.W.shape[0])


def sample_points(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 12345)

    x = -1 + 2 * torch.rand(N_INT, 1, generator=g)
    y = -1 + 2 * torch.rand(N_INT, 1, generator=g)
    X_int = torch.cat([x, y], dim=1)

    n_each = N_BND // 4

    y_left = -1 + 2 * torch.rand(n_each, 1, generator=g)
    left = torch.cat([-torch.ones(n_each, 1), y_left], dim=1)

    y_right = -1 + 2 * torch.rand(n_each, 1, generator=g)
    right = torch.cat([torch.ones(n_each, 1), y_right], dim=1)

    x_bottom = -1 + 2 * torch.rand(n_each, 1, generator=g)
    bottom = torch.cat([x_bottom, -torch.ones(n_each, 1)], dim=1)

    x_top = -1 + 2 * torch.rand(n_each, 1, generator=g)
    top = torch.cat([x_top, torch.ones(n_each, 1)], dim=1)

    X_bnd = torch.cat([left, right, bottom, top], dim=0)

    return X_int.to(DEVICE), X_bnd.to(DEVICE)


def interior_residual_vector(model, X_int):
    X = X_int.detach().clone().requires_grad_(True)

    out = model(X)
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    grad_u = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    grad_p = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True, retain_graph=True)[0]

    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

    grad_ux = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    grad_uy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    grad_vx = torch.autograd.grad(v_x, X, torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
    grad_vy = torch.autograd.grad(v_y, X, torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]

    u_xx, u_yy = grad_ux[:, 0:1], grad_uy[:, 1:2]
    v_xx, v_yy = grad_vx[:, 0:1], grad_vy[:, 1:2]

    f1, f2 = forcing_ns(X)

    mom_u = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) - f1
    mom_v = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) - f2
    div = u_x + v_y

    r = torch.cat([mom_u, mom_v, div], dim=0).reshape(-1)
    return r / math.sqrt(3 * N_INT)


def boundary_residual_vector(model, X_bnd):
    pred = model(X_bnd)
    u_true, v_true, _ = exact_uvp(X_bnd)

    ru = pred[:, 0:1] - u_true
    rv = pred[:, 1:2] - v_true

    r = torch.cat([ru, rv], dim=0).reshape(-1)
    return r / math.sqrt(2 * N_BND)


def component_losses(model, X_int, X_bnd):
    r_int = interior_residual_vector(model, X_int)
    r_bnd = boundary_residual_vector(model, X_bnd)
    li = 0.5 * torch.sum(r_int ** 2)
    lb = 0.5 * torch.sum(r_bnd ** 2)
    return li, lb


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def grad_norm_of_loss(loss, model):
    params = trainable_params(model)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    flat = flatten_grads(grads)
    return torch.linalg.norm(flat).detach()


def select_beta_gradnorm(model, li, lb):
    gi = grad_norm_of_loss(li, model)
    gb = grad_norm_of_loss(lb, model)
    beta = gb / (gi + gb + GRAD_EPS)
    return float(torch.clamp(beta, 0.0, 1.0).item())


def jacobian_residuals(model, residual_vec):
    params = trainable_params(model)
    rows = []

    for i in range(residual_vec.numel()):
        grads = torch.autograd.grad(
            residual_vec[i],
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        rows.append(flatten_grads(grads))

    return torch.stack(rows, dim=0)


@torch.no_grad()
def smallest_eigval_psd(M):
    M = 0.5 * (M + M.T)
    return torch.linalg.eigvalsh(M)[0].item()


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


@torch.no_grad()
def relative_l2_velocity(model):
    xs = torch.linspace(-1, 1, L2_GRID, device=DEVICE)
    ys = torch.linspace(-1, 1, L2_GRID, device=DEVICE)

    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

    pred = model(XY)
    u_true, v_true, _ = exact_uvp(XY)

    pred_vel = torch.cat([pred[:, 0:1], pred[:, 1:2]], dim=1).reshape(-1)
    true_vel = torch.cat([u_true, v_true], dim=1).reshape(-1)

    return (torch.linalg.norm(pred_vel - true_vel) / torch.linalg.norm(true_vel)).item()


# Two-pass EMA
def ema(y, alpha):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y

    fwd = np.zeros_like(y)
    fwd[0] = y[0]
    for i in range(1, len(y)):
        fwd[i] = alpha * y[i] + (1.0 - alpha) * fwd[i - 1]

    bwd = np.zeros_like(y)
    bwd[-1] = fwd[-1]
    for i in range(len(y) - 2, -1, -1):
        bwd[i] = alpha * fwd[i] + (1.0 - alpha) * bwd[i + 1]

    return bwd


def run_one(seed, mode="EW"):
    set_seed(seed)

    X_int, X_bnd = sample_points(seed)

    model = TwoLayerReLU3NS(WIDTH).to(DEVICE)
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

        if mode == "EW":
            beta = 0.5
        elif mode == "GradNorm":
            beta = select_beta_gradnorm(model, li, lb)
        elif mode == "ASW":
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss = beta * li + (1.0 - beta) * lb

        if k % LOG_EVERY == 0 or k == STEPS:
            l2 = relative_l2_velocity(model)

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
                beta=f"{beta:.3f}",
            )

        if k == STEPS:
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    hist["runtime_sec"] = time.time() - start
    return hist


all_results = {mode: [] for mode in MODES}

for seed in SEEDS:
    for mode in MODES:
        print(f"\n=== Seed {seed}: {mode} ===")
        all_results[mode].append(run_one(seed, mode))


def stack_metric(mode, key):
    return np.stack([np.asarray(r[key]) for r in all_results[mode]], axis=0)


steps = np.asarray(all_results["EW"][0]["steps"])

summary = {}
for mode in MODES:
    summary[mode] = {}
    for key in ["loss", "rel_l2", "interior", "boundary", "beta"]:
        arr = stack_metric(mode, key)
        summary[mode][key + "_mean"] = arr.mean(axis=0)
        summary[mode][key + "_std"] = arr.std(axis=0)


print("\nFINAL RESULTS")

for mode in MODES:
    l2 = stack_metric(mode, "rel_l2")[:, -1]
    loss = stack_metric(mode, "loss")[:, -1]
    interior = stack_metric(mode, "interior")[:, -1]
    boundary = stack_metric(mode, "boundary")[:, -1]

    print(f"\n{mode}")
    print(f"Final velocity relative L2: {l2.mean():.4f} ± {l2.std():.4f}")
    print(f"Final weighted loss:        {loss.mean():.4e} ± {loss.std():.4e}")
    print(f"Final interior loss:        {interior.mean():.4e} ± {interior.std():.4e}")
    print(f"Final boundary loss:        {boundary.mean():.4e} ± {boundary.std():.4e}")

ew_l2 = stack_metric("EW", "rel_l2")[:, -1]
asw_l2 = stack_metric("ASW", "rel_l2")[:, -1]
gn_l2 = stack_metric("GradNorm", "rel_l2")[:, -1]

print("\nRelative improvement vs EW:")
print(f"GradNorm: {(gn_l2.mean() / ew_l2.mean() - 1) * 100:.1f}%")
print(f"ASW:      {(asw_l2.mean() / ew_l2.mean() - 1) * 100:.1f}%")

print("\nWin rates vs EW:")
print(f"GradNorm: {(gn_l2 < ew_l2).sum()}/{len(SEEDS)}")
print(f"ASW:      {(asw_l2 < ew_l2).sum()}/{len(SEEDS)}")

print("\nASW vs GradNorm:")
print(f"ASW wins: {(asw_l2 < gn_l2).sum()}/{len(SEEDS)}")


save_dict = {"steps": steps}
for mode in MODES:
    prefix = mode.lower()
    save_dict[f"{prefix}_loss"] = stack_metric(mode, "loss")
    save_dict[f"{prefix}_rel_l2"] = stack_metric(mode, "rel_l2")
    save_dict[f"{prefix}_interior"] = stack_metric(mode, "interior")
    save_dict[f"{prefix}_boundary"] = stack_metric(mode, "boundary")
    save_dict[f"{prefix}_beta"] = stack_metric(mode, "beta")

np.savez(f"{OUTDIR}/ns_ew_gradnorm_asw_results_raw.npz", **save_dict)


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
})

# Plot 1: loss and relative L2
plt.figure(figsize=(11, 4.2))

plt.subplot(1, 2, 1)
for mode in MODES:
    plt.plot(
        steps,
        ema(summary[mode]["loss_mean"], EMA_ALPHA_LOSS),
        label=mode,
    )
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Weighted training loss")
plt.title("Navier-Stokes: Training loss")
plt.legend()
plt.grid(alpha=0.25)

plt.subplot(1, 2, 2)
for mode in MODES:
    plt.plot(
        steps,
        ema(summary[mode]["rel_l2_mean"], EMA_ALPHA_L2),
        label=mode,
    )
plt.xlabel("Iteration")
plt.ylabel("Velocity relative $L^2$ error")
plt.title("Navier-Stokes: Relative $L^2$ error")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_ns_loss_l2_ew_gradnorm_asw_smooth.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2: components
plt.figure(figsize=(11, 4.2))

plt.subplot(1, 2, 1)
for mode in MODES:
    plt.plot(
        steps,
        ema(summary[mode]["interior_mean"], EMA_ALPHA_COMPONENT),
        label=mode,
    )
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Interior residual loss")
plt.title("Navier-Stokes: Interior residual")
plt.legend()
plt.grid(alpha=0.25)

plt.subplot(1, 2, 2)
for mode in MODES:
    plt.plot(
        steps,
        ema(summary[mode]["boundary_mean"], EMA_ALPHA_COMPONENT),
        label=mode,
    )
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Boundary loss")
plt.title("Navier-Stokes: Boundary loss")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_ns_components_ew_gradnorm_asw_smooth.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3: beta trajectory
plt.figure(figsize=(6.5, 4.2))

for mode in MODES:
    plt.plot(
        steps,
        ema(summary[mode]["beta_mean"], EMA_ALPHA_BETA),
        label=mode,
    )

plt.xlabel("Iteration")
plt.ylabel("$\\beta$ on interior residual")
plt.title("Navier-Stokes: Weight trajectory")
plt.ylim(-0.02, 1.02)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig3_ns_beta_ew_gradnorm_asw_smooth.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figures and raw results to: {OUTDIR}/")
