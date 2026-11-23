"""Bütün deneyler -> sonuclar/ (28 süreçle dakikalar sürer)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import corels_py as cp
import veriler

warnings.filterwarnings("ignore")
SONUC = veriler.KOK / "sonuclar"
LAMLAR = [0.02, 0.01, 0.005]
AZAMI_SURE = 240


def katlar(n, tohum=0):
    return np.random.default_rng(tohum).permutation(n) % 10


def oranlar(y, t, m):
    return dict(TPR=t[m & (y == 1)].mean(), FPR=t[m & (y == 0)].mean())


def compas_kat(yas, irk, lam, k):
    d = veriler.compas()
    G, adlar, gr, y = veriler.compas_oznitelik(d, yas, irk)
    f = katlar(len(y))
    tr, te = f != k, f == k
    onc = cp.oncul_kazi(G[tr], adlar, gr)
    m = cp.coz(G[tr], y[tr], onc, lam, AZAMI_SURE, 4_000_000 if irk else 400_000)
    t, c = cp.tahmin(m, G), (d.decile_score.to_numpy() >= 5).astype(int)
    s = dict(yas=yas, irk=irk, lam=lam, kat=k, M=len(onc), dogruluk=(t[te] == y[te]).mean(),
             egitim=(t[tr] == y[tr]).mean(), compas=(c[te] == y[te]).mean(), kural=len(m["liste"]),
             varsayilan=m["varsayilan"], hepsi_evet=all(q for _, q in m["liste"]), kanit=m["kanit"],
             sure=m["sure"], gezilen=m["gezilen"], liste=cp.kimlik(m, adlar),
             sirali=cp.yaz(m, adlar).replace("\n", " | "))
    for kisa, r in (("siyah", "African-American"), ("beyaz", "Caucasian")):
        g = te & (d.race == r).to_numpy()
        for ad, v in (("CORELS", t), ("COMPAS", c)):
            for o, x in oranlar(y, v, g).items():
                s[f"{ad}_{kisa}_{o}"] = x
    return s


def ripper(G, y, tr, te, tohum):
    import wittgenstein as lw
    X = pd.DataFrame(G.astype(int), columns=[f"f{i}" for i in range(G.shape[1])])
    m = lw.RIPPER(random_state=tohum)
    m.fit(X[tr].assign(y=y[tr]), class_feat="y", pos_class=1)
    return np.asarray(m.predict(X[te])).astype(int)


MODELLER = {"Lojistik reg.": lambda: LogisticRegression(max_iter=1000), "SVM": lambda: SVC(),
            "AdaBoost": lambda: AdaBoostClassifier(algorithm="SAMME", random_state=0),
            "CART": lambda: DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=7, ccp_alpha=0.0025, random_state=0),
            "Rastgele orman": lambda: RandomForestClassifier(500, random_state=0)}


def rakip_kat(ad, k):
    G, _, _, y = veriler.compas_oznitelik(veriler.compas())
    f = katlar(len(y))
    tr, te = f != k, f == k
    t = ripper(G, y, tr, te, 0) if ad == "RIPPER" else MODELLER[ad]().fit(G[tr], y[tr]).predict(G[te])
    return dict(yontem=ad, kat=k, dogruluk=(t == y[te]).mean())


def bootstrap(b):
    G, adlar, gr, y = veriler.compas_oznitelik(veriler.compas())
    i = np.random.default_rng(b).integers(0, len(y), len(y)) if b >= 0 else np.arange(len(y))
    m = cp.coz(G[i], y[i], cp.oncul_kazi(G[i], adlar, gr), 0.005, AZAMI_SURE)
    return dict(ornek=b, kural=len(m["liste"]), kanit=m["kanit"], liste=cp.kimlik(m, adlar))


def kat697(yontem, k):
    d = veriler.oku697()
    y = (d.Target == "Dropout").astype(int).to_numpy()
    f = katlar(len(y))
    tr, te = f != k, f == k
    G, adlar, gr = veriler.oznitelik697(d, veriler.esikler697(d[tr]))     # eşikler eğitimden
    s = dict(yontem=yontem, kat=k)
    if yontem.startswith("CORELS"):
        m = cp.coz(G[tr], y[tr], cp.oncul_kazi(G[tr], adlar, gr, 0.01), float(yontem.split("=")[1]), AZAMI_SURE)
        t = cp.tahmin(m, G[te])
        s.update(kural=len(m["liste"]), kanit=m["kanit"], sure=m["sure"], liste=cp.kimlik(m, adlar))
    elif yontem == "Çoğunluk":
        t = np.full(te.sum(), int(y[tr].mean() >= .5))
    else:
        t = ripper(G, y, tr, te, 0) if yontem == "RIPPER" else MODELLER[yontem]().fit(G[tr], y[tr]).predict(G[te])
    s.update(dogruluk=(t == y[te]).mean(), TPR=t[y[te] == 1].mean(), FPR=t[y[te] == 0].mean())
    return s


def calistir(n_jobs=28, secim=("ana", "rakip", "bootstrap", "kendi", "irk")):
    SONUC.mkdir(exist_ok=True)
    veriler.compas(), veriler.oku697()
    P = Parallel(n_jobs=n_jobs, batch_size=1, verbose=1)
    is_ = {"ana": [delayed(compas_kat)(a, False, l, k) for l in reversed(LAMLAR) for a in ("yas_tarama", "yas_ham") for k in range(10)],
           "rakip": [delayed(rakip_kat)(a, k) for a in ["RIPPER", *MODELLER] for k in range(10)],
           "bootstrap": [delayed(bootstrap)(b) for b in range(-1, 100)],
           "kendi": [delayed(kat697)(a, k) for a in ["CORELS λ=0.02", "CORELS λ=0.03", "RIPPER", "CART", "Rastgele orman", "Lojistik reg.", "Çoğunluk"] for k in range(10)],
           "irk": [delayed(compas_kat)("yas_tarama", True, 0.005, k) for k in range(10)]}
    for ad in secim:
        pd.DataFrame(P(is_[ad])).to_csv(SONUC / f"{ad}.csv", index=False)
        print("bitti:", ad, flush=True)


if __name__ == "__main__":
    calistir(secim=sys.argv[1:] or ("ana", "rakip", "bootstrap", "kendi", "irk"))
