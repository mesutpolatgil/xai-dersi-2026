"""Bölüm 8 yeniden üretimi: doğruluk (5 bölme) ve yarı-sentetik sadakat deneyleri (~10-15 dk)."""
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import gam
import veriler

warnings.filterwarnings("ignore")
URETICILER = [gam.EBM, gam.XGBGAM, gam.Spline, gam.LR]


def dogruluk(ad, tohum):
    X, y = veriler.VERILER[ad]()
    X = X.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=tohum, stratify=y)
    satirlar = []
    for M in gam.MODELLER:
        m = M().fit(Xtr, ytr)
        satirlar.append(dict(veri=ad, tohum=tohum, yontem=M.ad, auc=roc_auc_score(yte, m.olasilik(Xte))))
    return satirlar


def sadakat(ad, Uretici, tohum=0):
    """Gerçek veriye bir 'üretici' GAM uydurulur; etiketler onun olasılıklarından yeniden çekilir.
    Sonra her GAM bu yarı-sentetik etiketlere uydurulur ve şekil fonksiyonları üreticininkilerle karşılaştırılır."""
    X, y = veriler.VERILER[ad]()
    X = X.to_numpy()
    ure = Uretici().fit(X, y)
    rng = np.random.default_rng(tohum)
    ys = (rng.random(len(X)) < ure.olasilik(X)).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, ys, test_size=0.3, random_state=tohum, stratify=ys)
    K_gercek = ure.katkilar(Xte)
    satirlar = []
    for M in gam.MODELLER:
        m = M().fit(Xtr, ytr)
        satirlar.append(dict(veri=ad, uretici=Uretici.ad, yontem=M.ad,
                             hata=gam.sadakat_hatasi(m.katkilar(Xte), K_gercek),
                             auc=roc_auc_score(yte, m.olasilik(Xte))))
    return satirlar


if __name__ == "__main__":
    os.makedirs("sonuclar", exist_ok=True)
    s1 = Parallel(n_jobs=-1, verbose=5)(delayed(dogruluk)(ad, t) for ad in veriler.VERILER for t in range(5))
    pd.DataFrame([r for b in s1 for r in b]).to_csv("sonuclar/dogruluk.csv", index=False)
    s2 = Parallel(n_jobs=-1, verbose=5)(delayed(sadakat)(ad, U) for ad in veriler.VERILER for U in URETICILER)
    pd.DataFrame([r for b in s2 for r in b]).to_csv("sonuclar/sadakat.csv", index=False)
    print("bitti")
