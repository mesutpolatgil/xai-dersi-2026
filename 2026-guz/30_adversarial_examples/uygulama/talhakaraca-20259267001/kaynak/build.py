"""Etkileşimli sayfanın verisini üretir ve template.html içine gömer.
Çalıştırma:  python build.py   (önce: python egit.py)
"""
import base64, json, sys, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from core import *
from cnn import CNN

HERE = Path(__file__).parent
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
Xtr8, ytr, Xte8, yte = veri(); Xtr = (Xtr8 / 255).astype(np.float32); Xte = (Xte8 / 255).astype(np.float32)

def yukle(ad):
    m = MLP(); m.W1, m.b1, m.W2, m.b2 = [v for v in np.load(HERE / ad).values()]; return m

def nicemle(m):
    """Ağırlıkları int16'ya indir; tarayıcıdaki modelle birebir aynı sayılar kullanılsın."""
    o = {"h": int(m.W1.shape[1])}
    for ad in ("W1", "W2"):
        W = getattr(m, ad); s = float(np.abs(W).max() / 32767); q = np.round(W / s).astype(np.int16)
        setattr(m, ad, (q * np.float32(s)).astype(np.float32)); o[ad] = b64(q); o["s" + ad[1]] = s
    o["b1"] = b64(m.b1.astype(np.float32)); o["b2"] = b64(m.b2.astype(np.float32)); return o

A, B = yukle("A.npz"), yukle("B.npz"); DA, DB = nicemle(A), nicemle(B)
acc = lambda m, X, y: float((m.predict(X) == y).mean())
print("A doğruluk %.4f · B doğruluk %.4f" % (acc(A, Xte, yte), acc(B, Xte, yte)))

# ---- eğriler (savunma grafiği) ----
T, yT = Xte[:2000], yte[:2000]; eps = [round(0.025 * i, 3) for i in range(13)]
curve = {"n": 2000, "eps": eps, "cleanA": acc(A, T, yT), "cleanB": acc(B, T, yT)}
for ad, m in (("A", A), ("B", B)):
    curve[ad + "_fgsm"] = [acc(m, fgsm(m, T, yT, e), yT) for e in eps]
    curve[ad + "_pgd"] = [acc(m, pgd(m, T, yT, e), yT) for e in eps]
print("eğri A fgsm", np.round(curve["A_fgsm"], 2)); print("eğri B fgsm", np.round(curve["B_fgsm"], 2)); print("eğri B pgd ", np.round(curve["B_pgd"], 2))

# ---- galeri: her sınıftan 3 görüntü ----
SH = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)]
def kararli(i):
    return all(A.predict(shift(Xte[i:i + 1], dx, dy))[0] == yte[i] for dx, dy in SH)
def tek_piksel(m, x, y):
    z1 = x @ m.W1 + m.b1
    for v in (0.0, 1.0):
        Z = np.maximum(z1[None] + (v - x)[:, None] * m.W1, 0) @ m.W2 + m.b2
        if (Z.argmax(1) != y).any(): return True
    return False
cand = [i for i in range(3000) if A.predict(Xte[i:i + 1])[0] == yte[i] and B.predict(Xte[i:i + 1])[0] == yte[i]]
kolay = [i for i in cand if tek_piksel(A, Xte[i], yte[i])]
print("1-piksel (kapsamlı arama, değer 0/1): %d/%d = %.1f%%" % (len(kolay), len(cand), 100 * len(kolay) / len(cand)))
gal = []
for k in range(10):
    e = [i for i in kolay if yte[i] == k][:1]
    n = [i for i in cand if yte[i] == k and i not in kolay and kararli(i)][:3 - len(e)]
    gal += n[:1] + e + n[1:]
gal_easy = [int(i in kolay) for i in gal]
print("galeri", len(gal), "kolay", sum(gal_easy))

# ---- kara kutu: rastgele orman ----
rf = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_leaf=2, n_jobs=2, random_state=42).fit(Xtr8[:30000], ytr[:30000])
off, feat, thr, left, right, cls = [], [], [], [], [], []
for e in rf.estimators_:
    t = e.tree_; off.append(len(feat))
    feat += [int(f) if f >= 0 else -1 for f in t.feature]; thr += [int(np.floor(v)) if f >= 0 else 0 for f, v in zip(t.feature, t.threshold)]
    left += [max(int(v), 0) for v in t.children_left]; right += [max(int(v), 0) for v in t.children_right]; cls += [int(v[0].argmax()) for v in t.value]
feat, thr, left, right, cls = np.array(feat, np.int16), np.array(thr, np.uint8), np.array(left, np.int16), np.array(right, np.int16), np.array(cls, np.uint8)
def rf_oy(X):
    out = []
    for x in X:
        v = np.zeros(10, int)
        for o in off:
            n = o
            while feat[n] >= 0: n = o + (left[n] if x[feat[n]] * 255 <= thr[n] + 0.5 else right[n])
            v[cls[n]] += 1
        out.append(v.argmax())
    return np.array(out)
ev = np.arange(5000, 5200); pool = np.arange(6000, 6400)
print("kara kutu (oy kuralı) doğruluk, 200 görüntü: %.3f · düğüm: %d" % ((rf_oy(Xte[ev]) == yte[ev]).mean(), len(feat)))
print("A, 200 görüntü: temiz %.3f · FGSM 0.1 %.3f" % (acc(A, Xte[ev], yte[ev]), acc(A, fgsm(A, Xte[ev], yte[ev], 0.1), yte[ev])))

# ---- yama modeli (küçük CNN) ve yama ----
C = CNN(G=1, F=32); C.Wc, C.bc, C.W1, C.b1, C.W2, C.b2 = [v for v in np.load(HERE / "C.npz").values()]
yama = json.loads((HERE / "yama.json").read_text()); P8 = np.load(HERE / yama["dosya"]); Pq = np.round(P8 * 255).astype(np.uint8)
DC = {"F": C.F, "H": C.H, "G": C.G, **{k: b64(getattr(C, k).astype(np.float32)) for k in ("Wc", "bc", "W1", "b1", "W2", "b2")}}
print("CNN doğruluk %.4f" % (C.predict(Xte) == yte).mean())

# ---- kitaptaki şekiller ----
import io
from PIL import Image
def sekil(dosya, cap):
    im = Image.open(HERE / "kitap-sekilleri" / dosya).convert("RGB"); im.thumbnail((760, 520)); b = io.BytesIO(); im.save(b, "JPEG", quality=82)
    return {"src": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(), "cap": cap}
FIGS = {"lbfgs": sekil("adversarial-ostrich.jpg", "Szegedy vd. Sol: doğru sınıf · Orta: eklenen hata · Sağ: hepsi “devekuşu” (CC-BY 3.0)"),
        "pixel": sekil("adversarial-1pixel.jpg", "Yalnızca daire içindeki piksel değişir; model yanlış sınıfı tahmin eder."),
        "patch": sekil("adversarial-toaster.jpg", "Brown vd.’nin yaması. Bir muzun yanına konunca model “tost makinesi” der."),
        "eot": sekil("adversarial-turtle.jpg", "Model, 3D baskılı kaplumbağayı “tüfek” olarak görür.")}

D = {"cls": SINIF, "figs": FIGS, "A": DA, "B": DB, "C": DC, "curve": curve,
     "stat": {"accA": acc(A, Xte, yte), "accB": acc(B, Xte, yte), "accC": float((C.predict(Xte) == yte).mean())},
     "gal": {"x": b64(Xte8[gal]), "y": [int(v) for v in yte[gal]], "easy": gal_easy},
     "ev": {"x": b64(Xte8[ev]), "y": [int(v) for v in yte[ev]]}, "pool": {"x": b64(Xte8[pool])},
     "rf": {"off": off, "feat": b64(feat), "thr": b64(thr), "left": b64(left), "right": b64(right), "cls": b64(cls)},
     "patch": {"k": int(Pq.shape[0]), "target": int(yama["hedef"]), "p": b64(Pq)}}
html = (HERE / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")))
out = HERE / (sys.argv[1] if len(sys.argv) > 1 else "kotucul-ornekler-laboratuvari.html"); out.write_text(html, encoding="utf-8")
print("yazıldı:", out, "%.2f MB" % (out.stat().st_size / 1e6))
