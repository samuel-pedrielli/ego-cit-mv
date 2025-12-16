import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# -------------------------
# Critic + revision operator (CIT)
# -------------------------
def critic_score(a, a0, welfare_min=0.5, k=8.0, sigma2=10.0):
    dist2 = torch.sum((a - a0) ** 2)
    close = torch.exp(-dist2 / sigma2)                 # long-range gradient
    welfare = torch.sigmoid(k * (a[0] - welfare_min))  # welfare gate
    return close * welfare

def revision_operator(a, a0, eps=1.0, welfare_min=0.5, k=8.0, sigma2=10.0):
    a_var = a.clone().detach().requires_grad_(True)
    C = critic_score(a_var, a0, welfare_min=welfare_min, k=k, sigma2=sigma2)
    C.backward()
    grad = a_var.grad.detach()
    grad = grad / (torch.norm(grad) + 1e-12)
    with torch.no_grad():
        a_tilde = a + eps * (1.0 - C.detach()) * grad
    return a_tilde, C.detach()

# -------------------------
# Task generator (multi-task)
# -------------------------
def make_task_b(seed, d=8, sigma=0.4):
    torch.manual_seed(seed)
    b = torch.zeros(d)
    b[0] = -1.0 - torch.abs(sigma * torch.randn(()))   # always conflicts with welfare
    b[1] =  2.0 + sigma * torch.randn(())
    b[2:] = torch.linspace(0.0, 1.0, d - 2) + sigma * torch.randn(d - 2)
    return b

# -------------------------
# Dynamics
# -------------------------
def run_trial_with_b(seed=0, b=None, d=8, steps=400, alpha=0.05, lam_cit=20.0, eps=1.0):
    torch.manual_seed(seed)
    a0 = torch.zeros(d); a0[0] = 1.0; a0[1] = 0.5
    a = a0.clone()

    hist = {"drift": [], "C": [], "task": [], "welfare": []}
    for _ in range(steps):
        gradE = (a - b)
        a_tilde, C = revision_operator(a, a0, eps=eps)

        with torch.no_grad():
            a = a - alpha * gradE - 2.0 * alpha * lam_cit * (a - a_tilde)

        hist["drift"].append(torch.norm(a - a0).item())
        hist["C"].append(C.item())
        hist["task"].append((-0.5 * torch.sum((a - b) ** 2)).item())
        hist["welfare"].append(a[0].item())
    return hist

def run_trial_l2_with_b(seed=0, b=None, d=8, steps=400, alpha=0.05, lam_l2=5.0):
    torch.manual_seed(seed)
    a0 = torch.zeros(d); a0[0] = 1.0; a0[1] = 0.5
    a = a0.clone()

    hist = {"drift": [], "C": [], "task": [], "welfare": []}
    for _ in range(steps):
        gradE = (a - b)
        _, C = revision_operator(a, a0, eps=1.0)

        with torch.no_grad():
            grad_l2 = 2.0 * lam_l2 * (a - a0)
            a = a - alpha * gradE - alpha * grad_l2

        hist["drift"].append(torch.norm(a - a0).item())
        hist["C"].append(C.item())
        hist["task"].append((-0.5 * torch.sum((a - b) ** 2)).item())
        hist["welfare"].append(a[0].item())
    return hist

# -------------------------
# Metrics
# -------------------------
def summarize_core(h):
    return {
        "final_welfare": float(h["welfare"][-1]),
        "final_task": float(h["task"][-1]),
        "final_drift": float(h["drift"][-1]),
        "final_critic": float(h["C"][-1]),
    }

def viol_metrics(h, thr):
    w = np.array(h["welfare"], dtype=float)
    viol = (w < thr).astype(float)
    viol_rate = float(viol.mean())
    viol_auc  = float(np.maximum(thr - w, 0.0).mean())
    return viol_rate, viol_auc

def mean_std(arr):
    arr = np.array(arr, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))

def eval_v5b(n_tasks=50, d=8, sigma_b=0.4, lam_l2=5.0, lam_cit=20.0, eps=1.0, thresholds=(0.5, 0.9, 0.95)):
    methods = ["NO_CIT", "L2", "CIT"]
    data = {m: {**{k: [] for k in ["final_welfare","final_task","final_drift","final_critic"]},
               **{f"viol_rate@{thr}": [] for thr in thresholds},
               **{f"viol_auc@{thr}": [] for thr in thresholds}} for m in methods}

    for s in range(n_tasks):
        b = make_task_b(s, d=d, sigma=sigma_b)

        h_no  = run_trial_with_b(seed=s, b=b, d=d, lam_cit=0.0,  eps=eps)
        h_l2  = run_trial_l2_with_b(seed=s, b=b, d=d, lam_l2=lam_l2)
        h_cit = run_trial_with_b(seed=s, b=b, d=d, lam_cit=lam_cit, eps=eps)

        for name, h in [("NO_CIT", h_no), ("L2", h_l2), ("CIT", h_cit)]:
            core = summarize_core(h)
            for k,v in core.items():
                data[name][k].append(v)
            for thr in thresholds:
                vr, va = viol_metrics(h, thr)
                data[name][f"viol_rate@{thr}"].append(vr)
                data[name][f"viol_auc@{thr}"].append(va)

    # Build summary table (mean±std)
    summary = {}
    for name in methods:
        summary[name] = {}
        for k in ["final_welfare","final_task","final_drift","final_critic"]:
            m,s = mean_std(data[name][k]); summary[name][k+"_mean"]=m; summary[name][k+"_std"]=s
        for thr in thresholds:
            m,s = mean_std(data[name][f"viol_rate@{thr}"]); summary[name][f"viol_rate@{thr}_mean"]=m; summary[name][f"viol_rate@{thr}_std"]=s
            m,s = mean_std(data[name][f"viol_auc@{thr}"]);  summary[name][f"viol_auc@{thr}_mean"]=m;  summary[name][f"viol_auc@{thr}_std"]=s

    return summary

def main():
    summary = eval_v5b()
    # Print in a readable form
    for name, dct in summary.items():
        print(name)
        for k in sorted(dct.keys()):
            print(f"  {k}: {dct[k]}")
        print()

    # Save a CSV-like text to results/
    import os
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "v5b_multitask_summary_from_script.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for name, dct in summary.items():
            f.write(name + "\n")
            for k in sorted(dct.keys()):
                f.write(f"{k},{dct[k]}\n")
            f.write("\n")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
