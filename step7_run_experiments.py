import pickle, os, sys, time, numpy as np
from n1 import *
OUT = "results.pkl"
res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else []
done = {(r["fd"], r["model"], r["seed"]) for r in res}
H = 64
def stride(X, y, k): return X[::k], y[::k]
def evaluate_snn(p, name, X, key):
    S = ENC[name](X, key); S = S if name == "direct" else S.astype(np.uint8)
    out, n1, n2 = [], [], []
    for i in range(0, len(S), 512):
        o, a, b = snn_apply(p, jnp.asarray(S[i:i+512], jnp.float32)); out.append(np.asarray(o)); n1.append(np.asarray(a)); n2.append(np.asarray(b))
    pred = np.clip(np.concatenate(out) * CAP, 0, CAP)
    nin = 0.0 if name == "direct" else S.reshape(len(S), -1).sum(1).mean()
    l1, l2 = np.concatenate(n1).mean(), np.concatenate(n2).mean()
    Fp = S.shape[-1]; steps = S.shape[1]
    macs = Fp * H * steps if name == "direct" else 0.0            # analog input -> first layer MACs
    synops = (0 if name == "direct" else nin * H) + l1 * H * 2 + l2 * 1   # W2 + recurrent R1 fan-out
    return pred, dict(in_spikes=nin, l1_spikes=l1, l2_spikes=l2, synops=synops, macs=macs,
                      energy_nJ=(synops * E_AC + macs * E_MAC) * 1e9, sparsity=1 - (nin / (Fp * steps)) if name != "direct" else 0.0)
def perturb(X, kind, seed):
    r = np.random.default_rng(seed); Y = X.copy()
    if kind == "noise": Y = np.clip(Y + r.normal(0, 0.05, Y.shape), 0, 1)
    if kind == "drop": Y[:, :, r.integers(Y.shape[2], size=1)] = 0.5      # one sensor stuck at mid-scale
    return Y.astype(np.float32)
jobs = []
for fd, st in [("FD001", 2), ("FD004", 6)]:
    for seed in [0, 1, 2]:
        for m in ["lstm", "cnn"] + [e for e in ENC if e != "step-forward"]:
            jobs.append((fd, m, seed, st))
for fd, m, seed, st in jobs:
    if (fd, m, seed) in done: continue
    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, yte = load(fd, seed=seed); Xtr, ytr = stride(Xtr, ytr, st)
    r = dict(fd=fd, model=m, seed=seed, n_train=len(Xtr))
    if m in ("lstm", "cnn"):
        p, pr, best, ne = train(m, Xtr, ytr, Xva, yva, seed)
        pred = np.clip(pr(p, Xte), 0, CAP); F = Xte.shape[2]
        macs = WIN * (F + 64) * 256 + 64 if m == "lstm" else WIN * 32 * (F * 5 + 32 * 5 * 2) + 32 * WIN
        r.update(macs=macs, energy_nJ=macs * E_MAC * 1e9, synops=0)
        for k in ("noise", "drop"):
            r["rmse_" + k] = rmse(np.clip(pr(p, perturb(Xte, k, seed)), 0, CAP), yte)
    else:
        f = ENC[m]; cast = (lambda a: a) if m == "direct" else (lambda a: a.astype(np.uint8))
        S, Sv = cast(f(Xtr, seed)), cast(f(Xva, seed + 100))
        p, pr, best, ne = train("snn", S, ytr, Sv, yva, seed)
        pred, stats = evaluate_snn(p, m, Xte, seed + 200); r.update(stats)
        rec = decode(m, f(Xte, seed + 200), Xte); r["recon_rmse"] = float(np.sqrt(((rec - Xte) ** 2).mean()))
        for k in ("noise", "drop"):
            r["rmse_" + k] = rmse(evaluate_snn(p, m, perturb(Xte, k, seed), seed + 200)[0], yte)
    ytrue = np.loadtxt(f"{DATA}/RUL_{fd}.txt")
    r.update(val_rmse=best, epochs=ne, rmse=rmse(pred, yte), nasa=nasa(pred, yte), time=time.time() - t0,
             rmse_uncapped=rmse(pred, ytrue), nasa_uncapped=nasa(pred, ytrue), pred=pred)
    res.append(r); pickle.dump(res, open(OUT, "wb"))
    print(fd, m, seed, "rmse %.2f nasa %.0f ep %d %.0fs" % (r["rmse"], r["nasa"], ne, r["time"]), flush=True)
print("ALL DONE")
