import math, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacrev

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

m            = 160
d_spatial    = 1
n1           = 96
n2           = 36
T_FINAL      = 1.0
X_LO, X_HI   = -1.0, 1.0
eta          = 5e-4          # was 1e-3
K            = 1500
T_UPDATE     = 25
N_SEEDS      = 3
GRID_LEVEL   = 20            # was 12 — finer simplex grid
ALPHA1_FLOOR = 0.05          # NEW: α₁ ≥ 0.05 (prevents PDE/bnd from being ignored)
RIDGE        = 1e-12
EVAL_EVERY   = 25
DIVERGE_TOL  = 1e15
EMA_ALPHA    = 0.05          # smoothing for display curves

PI    = math.pi
P_DIM = m * (d_spatial + 2)

def u_star(t, x):  return torch.exp(-t) * torch.sin(PI * x)
def f_forcing(t, x):
    et = torch.exp(-t); s = torch.sin(PI*x); c = torch.cos(PI*x)
    return -et*s + 6*(et*s)*(PI*et*c) + (-PI**3 * et * c)
-
def sig_act (z): return torch.tanh(z)
def sig_actp(z): return 1.0 - torch.tanh(z)**2

def sample_collocation(seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    t_int = torch.rand(n1, generator=g, dtype=torch.float64) * T_FINAL
    x_int = torch.rand(n1, generator=g, dtype=torch.float64) * (X_HI-X_LO) + X_LO
    n_e = n2 // 3
    t_b = torch.cat([torch.zeros(n_e, dtype=torch.float64),
                     torch.rand(n_e, generator=g, dtype=torch.float64) * T_FINAL,
                     torch.rand(n_e, generator=g, dtype=torch.float64) * T_FINAL])
    x_b = torch.cat([torch.rand(n_e, generator=g, dtype=torch.float64)*(X_HI-X_LO)+X_LO,
                     torch.full((n_e,), X_LO, dtype=torch.float64),
                     torch.full((n_e,), X_HI, dtype=torch.float64)])
    return (t_int.to(device), x_int.to(device), t_b.to(device), x_b.to(device))

def init_params(seed):
    g = torch.Generator(device='cpu').manual_seed(seed + 1234)
    theta = torch.randn(m, d_spatial+2, generator=g, dtype=torch.float64).to(device)
    a_u = (2*torch.randint(0,2,(m,),generator=g)-1).double().to(device)
    a_v = (2*torch.randint(0,2,(m,),generator=g)-1).double().to(device)
    a_w = (2*torch.randint(0,2,(m,),generator=g)-1).double().to(device)
    return theta, a_u, a_v, a_w

def forward(theta, a_u, a_v, a_w, t, x):
    x_aug = torch.stack([t, x, torch.ones_like(t)], dim=1)
    z = x_aug @ theta.T
    sig, sigp = sig_act(z), sig_actp(z)
    inv_sm = 1.0 / math.sqrt(m)
    u = inv_sm*(sig @ a_u); v = inv_sm*(sig @ a_v); w = inv_sm*(sig @ a_w)
    th_t, th_x = theta[:, 0], theta[:, 1]
    sigp_t = sigp * th_t.unsqueeze(0); sigp_x = sigp * th_x.unsqueeze(0)
    u_t = inv_sm*(sigp_t @ a_u); u_x = inv_sm*(sigp_x @ a_u)
    v_x = inv_sm*(sigp_x @ a_v); w_x = inv_sm*(sigp_x @ a_w)
    return u, v, w, u_t, u_x, v_x, w_x

def residuals(theta, a_u, a_v, a_w, t_int, x_int, t_bnd, x_bnd):
    u_i,v_i,w_i,u_t_i,u_x_i,v_x_i,w_x_i = forward(theta,a_u,a_v,a_w,t_int,x_int)
    f_i = f_forcing(t_int, x_int)
    s1 = (u_t_i + 6*u_i*v_i + w_x_i - f_i) / math.sqrt(n1)
    s2 = (v_i - u_x_i) / math.sqrt(n1)
    s3 = (w_i - v_x_i) / math.sqrt(n1)
    u_b, *_ = forward(theta, a_u, a_v, a_w, t_bnd, x_bnd)
    h  = (u_b - u_star(t_bnd, x_bnd)) / math.sqrt(n2)
    return torch.cat([s1, h]), s2, s3

def weighted_loss(theta, a_u, a_v, a_w, pts, alpha):
    r1, r2, r3 = residuals(theta, a_u, a_v, a_w, *pts)
    L = 0.5 * (alpha[0]*(r1**2).sum() + alpha[1]*(r2**2).sum() + alpha[2]*(r3**2).sum())
    return L, (r1, r2, r3)

def jacobian_blocks(theta, a_u, a_v, a_w, pts):
    def res_concat(th):
        r1, r2, r3 = residuals(th, a_u, a_v, a_w, *pts)
        return torch.cat([r1, r2, r3])
    Jf = jacrev(res_concat)(theta).reshape(-1, P_DIM).T
    d1, d2 = n1+n2, n1
    return Jf[:, :d1], Jf[:, d1:d1+d2], Jf[:, d1+d2:]

def simplex_grid(level, floor_alpha1):
    floor_idx = int(math.ceil(floor_alpha1 * level))
    pts = [[i/level, j/level, (level-i-j)/level]
           for i in range(floor_idx, level+1)
           for j in range(level+1-i)]
    return torch.tensor(pts, dtype=torch.float64, device=device)

def asw_select(D1, D2, D3, alpha_grid, chunk=8):
    D_full = torch.cat([D1, D2, D3], dim=1)
    J1, J2, J3 = D_full.T @ D1, D_full.T @ D2, D_full.T @ D3
    G1, G2, G3 = J1 @ J1.T, J2 @ J2.T, J3 @ J3.T
    Dt  = G1.shape[0]
    eye = torch.eye(Dt, dtype=torch.float64, device=device) * RIDGE
    best_lam, best_alpha = -float('inf'), alpha_grid[0]
    for i0 in range(0, alpha_grid.shape[0], chunk):
        ck = alpha_grid[i0:i0+chunk]
        a1, a2, a3 = ck[:,0:1,None], ck[:,1:2,None], ck[:,2:3,None]
        Gb  = a1*G1 + a2*G2 + a3*G3 + eye
        Gb  = 0.5*(Gb + Gb.transpose(-1,-2))
        lam = torch.linalg.eigvalsh(Gb)[:, 0]
        idx = torch.argmax(lam).item()
        if lam[idx].item() > best_lam:
            best_lam, best_alpha = lam[idx].item(), ck[idx].clone()
    return best_alpha, best_lam

def relative_l2_u(theta, a_u, a_v, a_w, n_grid=101):
    t_g = torch.linspace(0, T_FINAL, n_grid, dtype=torch.float64, device=device)
    x_g = torch.linspace(X_LO, X_HI, n_grid, dtype=torch.float64, device=device)
    Tg, Xg = torch.meshgrid(t_g, x_g, indexing='ij')
    with torch.no_grad():
        u_p, *_  = forward(theta, a_u, a_v, a_w, Tg.flatten(), Xg.flatten())
        u_t_true = u_star(Tg.flatten(), Xg.flatten())
        return (torch.norm(u_p - u_t_true) / (torch.norm(u_t_true)+1e-12)).item()

def train(method, seed):
    theta, a_u, a_v, a_w = init_params(seed)
    pts  = sample_collocation(seed)
    grid = simplex_grid(GRID_LEVEL, ALPHA1_FLOOR) if method == 'asw' else None
    alpha = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float64, device=device)
    theta = theta.detach().clone().requires_grad_(True)
    hist  = {'loss':[], 'r1':[], 'r2':[], 'r3':[], 'alpha':[], 'rel_l2':[]}
    diverged = False
    t0 = time.time()
    for k in range(K):
        if method == 'asw' and (k % T_UPDATE == 0):
            with torch.no_grad():
                D1, D2, D3 = jacobian_blocks(theta.detach(), a_u, a_v, a_w, pts)
                alpha, _   = asw_select(D1, D2, D3, grid)
        L, (r1, r2, r3) = weighted_loss(theta, a_u, a_v, a_w, pts, alpha)
        if (not torch.isfinite(L).item()) or L.item() > DIVERGE_TOL:
            print(f"  [{method}] seed {seed} diverged at iter {k}"); diverged=True; break
        grad = torch.autograd.grad(L, theta)[0]
        with torch.no_grad():
            theta.sub_(eta * grad)
        hist['loss' ].append(L.item())
        hist['r1'   ].append((r1.detach()**2).sum().item())
        hist['r2'   ].append((r2.detach()**2).sum().item())
        hist['r3'   ].append((r3.detach()**2).sum().item())
        hist['alpha'].append(alpha.detach().cpu().numpy().copy())
        if k % EVAL_EVERY == 0 or k == K-1:
            hist['rel_l2'].append((k, relative_l2_u(theta, a_u, a_v, a_w)))
    if not diverged:
        print(f"  [{method:>3s}] seed {seed} done in {time.time()-t0:5.1f}s  "
              f"final rel L² = {hist['rel_l2'][-1][1]:.4f}")
    hist['diverged'] = diverged
    return hist

results = {'ew': [], 'asw': []}
for seed in range(N_SEEDS):
    print(f"\n=== seed {seed} ===")
    results['ew' ].append(train('ew',  seed))
    results['asw'].append(train('asw', seed))

fl_ew  = np.array([h['rel_l2'][-1][1] for h in results['ew' ]])
fl_asw = np.array([h['rel_l2'][-1][1] for h in results['asw']])
print("\n" + "="*64)
print(f"KdV (J=3) ASW vs EW summary over {N_SEEDS} paired seeds")
print("="*64)
print(f"  EW  final rel L²: {fl_ew.mean():.4f} ± {fl_ew.std():.4f}   per-seed: {fl_ew}")
print(f"  ASW final rel L²: {fl_asw.mean():.4f} ± {fl_asw.std():.4f}   per-seed: {fl_asw}")
print(f"  ASW wins (paired): {int((fl_asw < fl_ew).sum())}/{N_SEEDS}")
print(f"  Mean reduction (rel L²): {(fl_ew.mean()-fl_asw.mean())/fl_ew.mean()*100:+.1f}%")

def ema(x, a=EMA_ALPHA):
    x = np.asarray(x, dtype=float); out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a*x[i] + (1-a)*out[i-1]
    return out

def mean_curve(runs, key):  return np.mean([r[key] for r in runs], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(ema(mean_curve(results['ew' ], 'loss')), label='EW',  color='C0', lw=1.8)
ax.plot(ema(mean_curve(results['asw'], 'loss')), label='ASW', color='C1', lw=1.8)
ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('weighted training loss')
ax.set_title('KdV (J=3) — convergence'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ks = [k for k,_ in results['ew'][0]['rel_l2']]
m_ew  = ema(np.mean([[v for _,v in r['rel_l2']] for r in results['ew' ]], axis=0))
m_asw = ema(np.mean([[v for _,v in r['rel_l2']] for r in results['asw']], axis=0))
ax.plot(ks, m_ew,  label='EW',  color='C0', lw=1.8)
ax.plot(ks, m_asw, label='ASW', color='C1', lw=1.8)
ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('relative L² error of u')
ax.set_title('KdV (J=3) — solution accuracy'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
alpha_mean = np.mean([h['alpha'] for h in results['asw']], axis=0)
for i, (lbl, c) in enumerate(zip(['α₁ (PDE+bnd)', 'α₂ (v−u_x)', 'α₃ (w−v_x)'],
                                 ['C0','C2','C3'])):
    ax.plot(ema(alpha_mean[:, i]), label=lbl, color=c, lw=1.8)
ax.axhline(1/3, color='gray', ls='--', alpha=0.7, label='Uniform (1/3)')
ax.axhline(ALPHA1_FLOOR, color='C0', ls=':', alpha=0.5, label=f'α₁ floor ({ALPHA1_FLOOR})')
ax.set_xlabel('iteration'); ax.set_ylabel('α component')
ax.set_title('ASW weight trajectory'); ax.set_ylim(-0.02, 1.02); ax.legend(loc='upper right'); ax.grid(alpha=0.3)

ax = axes[1, 1]
for label, key, c in [('r₁ (PDE+bnd)','r1','C0'), ('r₂ (v−u_x)','r2','C2'), ('r₃ (w−v_x)','r3','C3')]:
    ax.plot(ema(mean_curve(results['ew' ], key)), color=c, ls='--', alpha=0.7, lw=1.5, label=f'{label} EW')
    ax.plot(ema(mean_curve(results['asw'], key)), color=c, ls='-',  lw=1.8,        label=f'{label} ASW')
ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel(r'$\|r_j\|^2$')
ax.set_title('per-objective residuals'); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('kdv_j3_asw_vs_ew.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved figure to kdv_j3_asw_vs_ew.png")
