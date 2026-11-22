import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from core import Adam

class CNN:
    """conv 5x5 (F filtre) → ReLU → 3x3 bölgede max havuzlama (8x8 bölgeler) → FC(9F→H) → ReLU → FC(H→10)"""
    def __init__(s, F=16, H=64, seed=0, G=3):
        r = np.random.default_rng(seed); s.F, s.H, s.G = F, H, G; s.Q = G * G; s.S = 24 // G
        s.Wc = (r.standard_normal((25, F)) * np.sqrt(2 / 25)).astype(np.float32); s.bc = np.zeros(F, np.float32)
        s.W1 = (r.standard_normal((G * G * F, H)) * np.sqrt(2 / (G * G * F))).astype(np.float32); s.b1 = np.zeros(H, np.float32)
        s.W2 = (r.standard_normal((H, 10)) * np.sqrt(2 / H)).astype(np.float32); s.b2 = np.zeros(10, np.float32)
    def params(s): return [s.Wc, s.bc, s.W1, s.b1, s.W2, s.b2]
    def _fw(s, X):
        N = len(X); cols = swv(X.reshape(N, 28, 28), (5, 5), axis=(1, 2)).reshape(N, 576, 25)
        Z = np.maximum(cols @ s.Wc + s.bc, 0)                       # (N,576,F)
        R = Z.reshape(N, s.G, s.S, s.G, s.S, s.F).transpose(0, 1, 3, 2, 4, 5).reshape(N, s.Q, s.S * s.S, s.F)
        am = R.argmax(2); feat = np.take_along_axis(R, am[:, :, None, :], 2)[:, :, 0, :].reshape(N, s.Q * s.F)
        z1 = feat @ s.W1 + s.b1; a1 = np.maximum(z1, 0); return cols, am, feat, z1, a1, a1 @ s.W2 + s.b2
    def logits(s, X): return s._fw(X)[-1]
    def predict(s, X): return np.concatenate([s.logits(X[i:i + 2000]).argmax(1) for i in range(0, len(X), 2000)])
    def grads(s, X, y, want_params=True):
        N = len(X); cols, am, feat, z1, a1, z = s._fw(X)
        z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True); d = p; d[np.arange(N), y] -= 1
        dz1 = (d @ s.W2.T) * (z1 > 0); dfeat = (dz1 @ s.W1.T).reshape(N, s.Q, s.F) * (feat.reshape(N, s.Q, s.F) > 0)
        dR = np.zeros((N, s.Q, s.S * s.S, s.F), np.float32); np.put_along_axis(dR, am[:, :, None, :], dfeat[:, :, None, :], 2)
        dZ = dR.reshape(N, s.G, s.G, s.S, s.S, s.F).transpose(0, 1, 3, 2, 4, 5).reshape(N, 576, s.F)
        dcols = (dZ @ s.Wc.T).reshape(N, 24, 24, 25); gX = np.zeros((N, 28, 28), np.float32)
        for i in range(5):
            for j in range(5): gX[:, i:i + 24, j:j + 24] += dcols[..., i * 5 + j]
        gX = gX.reshape(N, 784)
        if not want_params: return None, gX
        return [np.einsum('nki,nkf->if', cols, dZ) / N, dZ.sum((0, 1)) / N, feat.T @ dz1 / N, dz1.mean(0), a1.T @ d / N, d.mean(0)], gX

def egit_cnn(X, y, epochs=6, seed=0, lr=2e-3, bs=128, **kw):
    rng = np.random.default_rng(seed); m = CNN(seed=seed, **kw); opt = Adam(m.params(), lr)
    for ep in range(epochs):
        if ep == epochs - 2: opt.lr = lr / 4
        perm = rng.permutation(len(X))
        for i in range(0, len(X), bs):
            gs, _ = m.grads(X[perm[i:i + bs]], y[perm[i:i + bs]]); opt.step(gs)
    return m
