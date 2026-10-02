import pickle, numpy as np, collections, json
from scipy import stats
R = pickle.load(open('results.pkl', 'rb'))
M = ["lstm", "cnn", "rate", "ttfs", "population", "send-on-delta", "sod-anchored", "level-crossing", "direct"]
by = collections.defaultdict(list)
for r in R: by[(r['fd'], r['model'])].append(r)
S = {}
for fd in ["FD001", "FD004"]:
    for m in M:
        rs = by[(fd, m)]
        if not rs: continue
        g = lambda k: np.array([r[k] for r in rs if k in r], float)
        S[(fd, m)] = {k: (g(k).mean(), g(k).std(), len(g(k))) for k in
                      ["rmse", "nasa", "rmse_uncapped", "nasa_uncapped", "rmse_noise", "rmse_drop", "energy_nJ", "synops", "macs",
                       "in_spikes", "l1_spikes", "l2_spikes", "sparsity", "recon_rmse", "epochs"] if len(g(k))}
        S[(fd, m)]['per_seed'] = [round(r['rmse'], 2) for r in sorted(rs, key=lambda r: r['seed'])]
for (fd, m), v in S.items():
    print(fd, f"{m:15s}", "rmse %.2f±%.2f n=%d" % v['rmse'], "nasa %.0f" % v['nasa'][0],
          "unc %.2f" % v['rmse_uncapped'][0] if 'rmse_uncapped' in v else "", "E %.1fnJ" % v['energy_nJ'][0],
          "in %.0f" % v['in_spikes'][0] if 'in_spikes' in v else "", "recon %.3f" % v['recon_rmse'][0] if 'recon_rmse' in v else "",
          "noise %.2f drop %.2f" % (v['rmse_noise'][0], v['rmse_drop'][0]), v['per_seed'])
# ranks over the 6 (dataset,seed) cells among SNN encodings, Friedman
enc = M[2:]
cells = []
for fd in ["FD001", "FD004"]:
    for s in [0, 1, 2]:
        row = []
        for m in enc:
            x = [r['rmse'] for r in by[(fd, m)] if r['seed'] == s]
            row.append(x[0] if x else np.nan)
        if not np.isnan(row).any(): cells.append(row)
cells = np.array(cells)
if len(cells) >= 2:
    rk = np.array([stats.rankdata(c) for c in cells]); print("ranks", dict(zip(enc, rk.mean(0).round(2))), "n cells", len(cells))
    print("friedman", stats.friedmanchisquare(*cells.T))
    a = enc.index("sod-anchored"); b = enc.index("send-on-delta")
    print("anchored better than plain in", int((cells[:, a] < cells[:, b]).sum()), "of", len(cells), "wilcoxon p", stats.wilcoxon(cells[:, a], cells[:, b]).pvalue)
pickle.dump(S, open('summary.pkl', 'wb'))
