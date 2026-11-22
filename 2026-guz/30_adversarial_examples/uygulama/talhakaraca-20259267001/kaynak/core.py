import gzip, urllib.request, numpy as np
from pathlib import Path
V = Path(__file__).parent / "veri"
URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
SINIF = ["Tişört","Pantolon","Kazak","Elbise","Mont","Sandalet","Gömlek","Spor ayakkabı","Çanta","Bot"]

def oku(ad, img):
    yol = V / f"{ad}.gz"
    if not yol.exists():   # veri yoksa indirilir (toplam yaklaşık 30 MB)
        V.mkdir(exist_ok=True); print("indiriliyor:", ad); urllib.request.urlretrieve(URL + f"{ad}.gz", yol)
    with gzip.open(yol, "rb") as f:
        d = np.frombuffer(f.read(), np.uint8, offset=16 if img else 8)
    return d.reshape(-1, 784) if img else d

def veri():
    return (oku("train-images-idx3-ubyte", True), oku("train-labels-idx1-ubyte", False).astype(np.int64),
            oku("t10k-images-idx3-ubyte", True), oku("t10k-labels-idx1-ubyte", False).astype(np.int64))

def shift(X, dx, dy):
    """X: (N,784) -> dx sağa, dy aşağı kaydır, boşluk 0."""
    I = X.reshape(-1, 28, 28); O = np.zeros_like(I)
    xs0, xs1 = max(0, dx), 28 + min(0, dx); ys0, ys1 = max(0, dy), 28 + min(0, dy)
    O[:, ys0:ys1, xs0:xs1] = I[:, ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return O.reshape(-1, 784)

def unshift_grad(G, dx, dy):
    return shift(G, -dx, -dy)

class MLP:
    def __init__(s, h=128, seed=0):
        r = np.random.default_rng(seed)
        s.W1 = (r.standard_normal((784, h)) * np.sqrt(2 / 784)).astype(np.float32); s.b1 = np.zeros(h, np.float32)
        s.W2 = (r.standard_normal((h, 10)) * np.sqrt(2 / h)).astype(np.float32); s.b2 = np.zeros(10, np.float32)
    def params(s): return [s.W1, s.b1, s.W2, s.b2]
    def logits(s, X):
        return np.maximum(X @ s.W1 + s.b1, 0) @ s.W2 + s.b2
    def proba(s, X):
        z = s.logits(X); z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    def predict(s, X): return s.logits(X).argmax(1)
    def grads(s, X, y, want_params=True):
        """CE kaybı; döner: (param gradyanları, girdi gradyanı (örnek başına, ölçeklenmemiş))"""
        z1 = X @ s.W1 + s.b1; a1 = np.maximum(z1, 0); z = a1 @ s.W2 + s.b2
        z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        d = p.copy(); d[np.arange(len(y)), y] -= 1
        dz1 = (d @ s.W2.T) * (z1 > 0)
        gX = dz1 @ s.W1.T
        if not want_params: return None, gX
        n = len(y)
        return [X.T @ dz1 / n, dz1.mean(0), a1.T @ d / n, d.mean(0)], gX

class Adam:
    def __init__(s, ps, lr=1e-3):
        s.ps = ps; s.lr = lr; s.m = [np.zeros_like(p) for p in ps]; s.v = [np.zeros_like(p) for p in ps]; s.t = 0
    def step(s, gs):
        s.t += 1
        for p, g, m, v in zip(s.ps, gs, s.m, s.v):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            p -= s.lr * (m / (1 - 0.9 ** s.t)) / (np.sqrt(v / (1 - 0.999 ** s.t)) + 1e-8)

def fgsm(model, X, y, eps):
    _, g = model.grads(X, y, False)
    return np.clip(X + eps * np.sign(g), 0, 1).astype(np.float32)

def pgd(model, X, y, eps, steps=10):
    Xa = X.copy(); a = eps / 4
    for _ in range(steps):
        Xa = fgsm(model, Xa, y, a); Xa = np.clip(np.minimum(np.maximum(Xa, X - eps), X + eps), 0, 1)
    return Xa

def aug(Xb, rng, m=2):
    out = np.empty_like(Xb); dxs = rng.integers(-m, m + 1, len(Xb)); dys = rng.integers(-m, m + 1, len(Xb))
    for dx in range(-m, m + 1):
        for dy in range(-m, m + 1):
            k = (dxs == dx) & (dys == dy)
            if k.any(): out[k] = shift(Xb[k], dx, dy)
    return out

def egit(X, y, h=128, epochs=12, adv_eps=None, seed=0, lr=1e-3, bs=128, augment=True):
    rng = np.random.default_rng(seed); m = MLP(h, seed); opt = Adam(m.params(), lr)
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        if ep == epochs - 3: opt.lr = lr / 4
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]; xb, yb = X[idx], y[idx]
            if augment: xb = aug(xb, rng)
            if adv_eps is not None:
                xb = np.concatenate([xb, fgsm(m, xb, yb, adv_eps)]); yb = np.concatenate([yb, yb])
            gs, _ = m.grads(xb, yb); opt.step(gs)
    return m
