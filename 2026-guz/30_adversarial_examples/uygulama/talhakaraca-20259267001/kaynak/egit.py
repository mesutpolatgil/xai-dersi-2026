"""Modelleri ve yamayı eğitir. Çalıştırma: python egit.py  (yaklaşık 6 dakika, GPU gerekmez)
A.npz : hedef model (784 → 128 → 10), ±2 piksel kaydırma artırmasıyla eğitilir
B.npz : aynı mimari, adversarial eğitim (her yığına FGSM ε = 0,1 örnekleri eklenir)
C.npz : yama deneyi için küçük evrişimli ağ (conv 5×5 → ReLU → global max → 64 → 10)
yama.npy, yama.json : C modeline karşı eğitilmiş evrensel yama
"""
import json, sys, numpy as np
from pathlib import Path
from core import *
from cnn import CNN, egit_cnn

HERE = Path(__file__).parent
Xtr8, ytr, Xte8, yte = veri(); Xtr = (Xtr8 / 255).astype(np.float32); Xte = (Xte8 / 255).astype(np.float32)
acc = lambda m, X, y: float((m.predict(X) == y).mean())
yeniden = "--hepsi" in sys.argv

if yeniden or not (HERE / "A.npz").exists():
    A = egit(Xtr, ytr, epochs=18); np.savez(HERE / "A.npz", *A.params()); print("A doğruluk %.4f" % acc(A, Xte, yte))
if yeniden or not (HERE / "B.npz").exists():
    B = egit(Xtr, ytr, epochs=18, adv_eps=0.1, seed=1); np.savez(HERE / "B.npz", *B.params()); print("B doğruluk %.4f" % acc(B, Xte, yte))
if yeniden or not (HERE / "C.npz").exists():
    C = egit_cnn(Xtr, ytr, epochs=8, G=1, F=32); np.savez(HERE / "C.npz", *C.params())
else:
    C = CNN(G=1, F=32); C.Wc, C.bc, C.W1, C.b1, C.W2, C.b2 = [v for v in np.load(HERE / "C.npz").values()]
print("C doğruluk %.4f" % acc(C, Xte, yte), flush=True)

def yapistir(X, P, r, c):
    k = P.shape[0]; I = X.reshape(-1, 28, 28).copy(); I[:, r:r + k, c:c + k] = P; return I.reshape(-1, 784)
def yama_egit(m, k, hedef, steps=800, seed=0, lr=0.05):
    rng = np.random.default_rng(seed); P = rng.uniform(0.3, 0.7, (k, k)).astype(np.float32); opt = Adam([P], lr); havuz = np.where(ytr != hedef)[0]
    for s in range(steps):
        g = np.zeros_like(P)
        for _ in range(4):   # her adımda 4 rastgele konum × 32 rastgele görüntü
            idx = rng.choice(havuz, 32); r, c = rng.integers(0, 29 - k, 2)
            _, gX = m.grads(yapistir(Xtr[idx], P, r, c), np.full(32, hedef), False); g += gX.reshape(-1, 28, 28)[:, r:r + k, c:c + k].mean(0)
        opt.step([g / 4]); np.clip(P, 0, 1, out=P)
    return P
def yama_olc(m, P, hedef, seed=5):
    rng = np.random.default_rng(seed); k = P.shape[0]; idx = np.where(yte[:2000] != hedef)[0]; hit = tot = 0
    for _ in range(10):
        r, c = rng.integers(0, 29 - k, 2); hit += (m.predict(yapistir(Xte[idx], P, r, c)) == hedef).sum(); tot += len(idx)
    return hit / tot

HEDEF = 8   # Çanta
for k in (6, 7, 8):
    P = np.round(yama_egit(C, k, HEDEF) * 255) / 255; rnd = np.random.default_rng(1).uniform(0, 1, (k, k)).astype(np.float32)
    b, r = yama_olc(C, P.astype(np.float32), HEDEF), yama_olc(C, rnd, HEDEF); print("yama %d×%d → %s: başarı %.3f · rastgele yama %.3f" % (k, k, SINIF[HEDEF], b, r), flush=True)
    np.save(HERE / f"yama{k}.npy", P.astype(np.float32))
    if k == 7: (HERE / "yama.json").write_text(json.dumps({"dosya": "yama7.npy", "hedef": HEDEF, "basari": b, "rastgele": r}))
