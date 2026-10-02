"""N1: spike encodings for RUL prediction on C-MAPSS (JAX, CPU).
Data pipeline, seven encodings, decoders, LIF SNN, LSTM/CNN baselines, metrics."""
import time, numpy as np, jax, jax.numpy as jnp, optax
from functools import partial

DATA = "data"   # folder with the C-MAPSS text files (created by step2_download_data.py)
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]   # 14 informative sensors
SLOPE = 5.0
BETA2_INIT = 5.0         # sigmoid(5) = 0.993: slow second layer integrates the whole window
BETA_INIT = 3.0          # sigmoid(3) = 0.95 initial membrane decay, learned per neuron
WIN, CAP, T = 30, 125.0, 4                                      # window, RUL cap, spike steps per cycle

# ------------------------------------------------------------------ data
def _load(fn):
    a = np.loadtxt(fn)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:5], a[:, 5:]

def load(fd, seed=0, val_frac=0.2):
    uid, cyc, ops, sen = _load(f"{DATA}/train_{fd}.txt")
    tuid, tcyc, tops, tsen = _load(f"{DATA}/test_{fd}.txt")
    rul_test = np.loadtxt(f"{DATA}/RUL_{fd}.txt")
    X, tX = sen[:, [s - 1 for s in SENSORS]], tsen[:, [s - 1 for s in SENSORS]]
    multi = fd in ("FD002", "FD004")
    if multi:   # six operating conditions: normalise each sensor per condition (fit on train)
        key = lambda o: np.round(o[:, 0]).astype(int) * 1000 + np.round(o[:, 1] * 100).astype(int)
        k, tk = key(ops), key(tops)
        for c in np.unique(k):
            m, s = X[k == c].mean(0), X[k == c].std(0) + 1e-8
            X[k == c] = (X[k == c] - m) / s
            tX[tk == c] = (tX[tk == c] - m) / s
    lo, hi = X.min(0), X.max(0)            # min-max to [0,1] fitted on train
    X = (X - lo) / (hi - lo + 1e-8); tX = np.clip((tX - lo) / (hi - lo + 1e-8), 0, 1)
    # RUL targets (piecewise linear, cap 125)
    y = np.zeros(len(uid))
    for u in np.unique(uid):
        m = uid == u; y[m] = np.minimum(cyc[m].max() - cyc[m], CAP)
    rng = np.random.default_rng(seed); units = np.unique(uid)
    val_u = set(rng.choice(units, int(len(units) * val_frac), replace=False))
    def windows(Xa, ya, ua, sel):
        xs, ys = [], []
        for u in sel:
            m = ua == u; xu, yu = Xa[m], ya[m]
            for e in range(WIN, len(xu) + 1):
                xs.append(xu[e - WIN:e]); ys.append(yu[e - 1])
        return np.array(xs, np.float32), np.array(ys, np.float32)
    Xtr, ytr = windows(X, y, uid, [u for u in units if u not in val_u])
    Xva, yva = windows(X, y, uid, sorted(val_u))
    Xte = []                                # last window per test engine (pad short ones)
    for u in np.unique(tuid):
        xu = tX[tuid == u]
        if len(xu) < WIN: xu = np.vstack([np.repeat(xu[:1], WIN - len(xu), 0), xu])
        Xte.append(xu[-WIN:])
    return Xtr, ytr, Xva, yva, np.array(Xte, np.float32), np.minimum(rul_test, CAP).astype(np.float32)

# ------------------------------------------------------------------ encodings
# input x: (N, WIN, F) in [0,1]; output spikes: (N, WIN*T, F') float32 in {0,1}; "direct" returns analog current
def enc_rate(x, key):
    p = np.repeat(x, T, axis=1)
    return (np.random.default_rng(key).random(p.shape) < p).astype(np.float32)

def enc_ttfs(x, key=None):
    N, W, F = x.shape; s = np.zeros((N, W, T, F), np.float32)
    t = np.clip(np.floor((1 - x) * T), 0, T - 1).astype(int)
    idx = np.indices((N, W, F))
    s[idx[0], idx[1], t, idx[2]] = 1.0
    return s.reshape(N, W * T, F)

def enc_population(x, key, K=5, sigma=0.15):
    c = np.linspace(0, 1, K)
    resp = np.exp(-((x[..., None] - c) ** 2) / (2 * sigma ** 2))       # (N,W,F,K)
    p = np.repeat(resp.reshape(*x.shape[:2], -1), T, axis=1)
    return (np.random.default_rng(key).random(p.shape) < p).astype(np.float32)

def _delta_like(x, theta, mode):
    """send-on-delta: multiple spikes (up to T) per cycle for large changes;
    step-forward: at most one spike per cycle; baseline starts at first value."""
    N, W, F = x.shape; up = np.zeros((N, W, T, F), np.float32); dn = np.zeros_like(up)
    base = x[:, 0, :].copy()
    for w in range(1, W):
        d = x[:, w, :] - base
        if mode == "sod":
            n = np.clip(np.floor(np.abs(d) / theta), 0, T).astype(int)
            for k in range(T):
                fire = n > k
                up[:, w, k][fire & (d > 0)] = 1; dn[:, w, k][fire & (d < 0)] = 1
            base += np.sign(d) * n * theta
        else:  # step-forward
            u, v = d > theta, d < -theta
            up[:, w, 0][u] = 1; dn[:, w, 0][v] = 1
            base += theta * u - theta * v
    return np.concatenate([up, dn], axis=3).reshape(N, W * T, 2 * F)

def enc_sod(x, key=None, theta=0.05): return _delta_like(x, theta, "sod")
def enc_sf(x, key=None, theta=0.05): return _delta_like(x, theta, "sf")

def enc_sod_anchored(x, key=None, theta=0.05):
    """send-on-delta plus a one-off time-to-first-spike code of the absolute level in the first cycle,
    so the network receives the starting level that pure change coding discards."""
    N, W, F = x.shape
    anchor = np.zeros((N, W * T, F), np.float32)
    anchor[:, :T, :] = enc_ttfs(x[:, :1, :])
    return np.concatenate([_delta_like(x, theta, "sod"), anchor], axis=2)

def enc_level(x, key=None, L=8):
    """level-crossing against fixed amplitude levels; one spike per crossed level (up to T)."""
    N, W, F = x.shape; lv = np.floor(x * L).astype(int)
    up = np.zeros((N, W, T, F), np.float32); dn = np.zeros_like(up)
    dl = np.diff(lv, axis=1)
    for k in range(T):
        up[:, 1:, k][dl > k] = 1; dn[:, 1:, k][dl < -k] = 1
    return np.concatenate([up, dn], axis=3).reshape(N, W * T, 2 * F)

def enc_direct(x, key=None): return np.repeat(x, T, axis=1).astype(np.float32)

ENC = {"rate": enc_rate, "ttfs": enc_ttfs, "population": enc_population,
       "send-on-delta": enc_sod, "sod-anchored": enc_sod_anchored, "step-forward": enc_sf,
       "level-crossing": enc_level, "direct": enc_direct}

# decoders: reconstruct the per-cycle signal (N,W,F) from spikes -> information-loss measure
def decode(name, s, x):
    N, W, F = x.shape
    if name == "direct": return x
    if name == "rate": return s.reshape(N, W, T, F).mean(2)
    if name == "ttfs":
        t = s.reshape(N, W, T, F).argmax(2); return 1 - (t + 0.5) / T
    if name == "population":
        K = 5; c = np.linspace(0, 1, K); r = s.reshape(N, W, T, F, K).mean(2) + 1e-6
        return (r * c).sum(-1) / r.sum(-1)
    if name == "sod-anchored":
        s4 = s[..., :2 * F].reshape(N, W, T, 2 * F); a = s[:, :T, 2 * F:]
        x0 = 1 - (a.argmax(1) + 0.5) / T
        return x0[:, None, :] + 0.05 * np.cumsum(s4[..., :F].sum(2) - s4[..., F:].sum(2), 1)
    s4 = s.reshape(N, W, T, 2 * F); up, dn = s4[..., :F].sum(2), s4[..., F:].sum(2)
    if name == "level-crossing":
        lv0 = np.floor(x[:, :1] * 8)
        return (lv0 + np.cumsum(up - dn, 1) + 0.5) / 8
    theta = 0.05
    return x[:, :1] + theta * np.cumsum(up - dn, 1)

# ------------------------------------------------------------------ models
@jax.custom_jvp
def spike(v): return (v > 0).astype(jnp.float32)
@spike.defjvp
def _spike_jvp(p, t):
    v, = p; dv, = t
    return spike(v), dv / (1.0 + SLOPE * jnp.abs(v)) ** 2          # fast-sigmoid surrogate

def glorot(key, i, o): return jax.random.normal(key, (i, o)) * jnp.sqrt(2.0 / (i + o))

def init_snn(key, fin, H=64):
    k = jax.random.split(key, 3)
    return {"W1": glorot(k[0], fin, H), "b1": jnp.zeros(H), "W2": glorot(k[1], H, H), "b2": jnp.zeros(H),
            "R1": glorot(jax.random.fold_in(key, 7), H, H) * 0.5,
            "a1": jnp.full(H, BETA_INIT), "a2": jnp.full(H, BETA2_INIT),
            "W3": glorot(k[2], H, 1), "b3": jnp.zeros(1)}

def snn_apply(p, s, beta=0.9, beta_o=0.95):
    """s: (B, S, F'). Two LIF layers (subtractive reset), leaky non-spiking readout."""
    B = s.shape[0]; H = p["W1"].shape[1]
    b1, b2 = jax.nn.sigmoid(p["a1"]), jax.nn.sigmoid(p["a2"])
    def step(c, x_t):
        v1, v2, vo = c[0], c[1], c[2]
        z1p = c[3]
        v1 = b1 * v1 + x_t @ p["W1"] + z1p @ p["R1"] + p["b1"]; z1 = spike(v1 - 1.0); v1 = v1 - z1
        v2 = b2 * v2 + z1 @ p["W2"] + p["b2"]; z2 = spike(v2 - 1.0); v2 = v2 - z2
        vo = vo + (z2 @ p["W3"] + p["b3"])          # time-averaged readout over the whole window
        return (v1, v2, vo, z1), (z1.sum(-1), z2.sum(-1))
    c0 = (jnp.zeros((B, H)), jnp.zeros((B, H)), jnp.zeros((B, 1)), jnp.zeros((B, H)))
    (v1, v2, vo, _), (n1, n2) = jax.lax.scan(step, c0, jnp.swapaxes(s, 0, 1))
    return vo[:, 0] / s.shape[1], n1.sum(0), n2.sum(0)   # mean readout; spike counts per sample

def init_lstm(key, F, H=64):
    k = jax.random.split(key, 2)
    return {"W": glorot(k[0], F + H, 4 * H), "b": jnp.zeros(4 * H).at[H:2 * H].set(1.0),
            "Wo": glorot(k[1], H, 1), "bo": jnp.zeros(1)}

def lstm_apply(p, x):
    B = x.shape[0]; H = p["Wo"].shape[0]
    def step(c, x_t):
        h, cc = c; g = jnp.concatenate([x_t, h], -1) @ p["W"] + p["b"]
        i, f, o, u = jnp.split(g, 4, -1)
        cc = jax.nn.sigmoid(f) * cc + jax.nn.sigmoid(i) * jnp.tanh(u); h = jax.nn.sigmoid(o) * jnp.tanh(cc)
        return (h, cc), None
    (h, _), _ = jax.lax.scan(step, (jnp.zeros((B, H)), jnp.zeros((B, H))), jnp.swapaxes(x, 0, 1))
    return (h @ p["Wo"] + p["bo"])[:, 0]

def init_cnn(key, F, C=32):
    k = jax.random.split(key, 4)
    return {"K1": jax.random.normal(k[0], (C, F, 5)) * jnp.sqrt(2 / (F * 5)), "K2": jax.random.normal(k[1], (C, C, 5)) * jnp.sqrt(2 / (C * 5)),
            "K3": jax.random.normal(k[2], (C, C, 5)) * jnp.sqrt(2 / (C * 5)), "Wd": glorot(k[3], C * WIN, 1), "bd": jnp.zeros(1)}

def cnn_apply(p, x):
    h = jnp.swapaxes(x, 1, 2)                                  # (B,F,W)
    for k in ("K1", "K2", "K3"):
        h = jax.nn.relu(jax.lax.conv_general_dilated(h, p[k], (1,), "SAME", dimension_numbers=("NCH", "OIH", "NCH")))
    return (h.reshape(h.shape[0], -1) @ p["Wd"] + p["bd"])[:, 0]

# ------------------------------------------------------------------ metrics
def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2)))
def nasa(p, y):
    d = p - y; return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

E_AC, E_MAC = 0.9e-12, 4.6e-12      # 45 nm, 32-bit float add / multiply-accumulate (Horowitz, 2014)

# ------------------------------------------------------------------ training
def train(model, Xtr, ytr, Xva, yva, seed, epochs=50, bs=256, lr=3e-3, patience=8, enc=None):
    """model in {'snn','lstm','cnn'}; for snn, Xtr/Xva are already encoded."""
    key = jax.random.PRNGKey(seed); F = Xtr.shape[-1]
    if model == "snn": p = init_snn(key, F); f = lambda p, x: snn_apply(p, x)[0]
    elif model == "lstm": p = init_lstm(key, F); f = lstm_apply
    else: p = init_cnn(key, F); f = cnn_apply
    opt = optax.adam(lr); st = opt.init(p)
    @jax.jit
    def upd(p, st, x, y):
        l, g = jax.value_and_grad(lambda p: jnp.mean((f(p, x.astype(jnp.float32)) - y / CAP) ** 2))(p)
        u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    pred = jax.jit(lambda p, x: f(p, x.astype(jnp.float32)))
    def predict(p, X):
        out = []
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]; n = len(xb)
            if n < bs: xb = np.concatenate([xb, np.zeros((bs - n, *xb.shape[1:]), xb.dtype)])
            out.append(np.asarray(pred(p, xb))[:n])
        return np.concatenate(out) * CAP
    rng = np.random.default_rng(seed); best, bp, wait, hist = 1e9, p, 0, []
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr)); n = len(idx) // bs * bs
        for i in range(0, n, bs):
            b = idx[i:i + bs]; p, st, l = upd(p, st, Xtr[b], ytr[b])
        v = rmse(np.clip(predict(p, Xva), 0, CAP), yva); hist.append(v)
        if v < best - 1e-3: best, bp, wait = v, p, 0
        else:
            wait += 1
            if wait >= patience: break
    return bp, predict, best, len(hist)
