"""Makaledeki dört veri setinde 5 katlı çapraz doğrulama ile kıyaslama (~15-25 dk)."""
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

import puan

warnings.filterwarnings("ignore")
VERILER = ["mammo", "mushroom", "adult", "bank"]
KLER = [3, 5, 7, 10]


def is_(ds, kat, k):
    d = pd.read_csv(f"veri/{ds}_data.csv")
    y = d.iloc[:, 0].to_numpy()
    X = d.iloc[:, 1:].to_numpy(float)
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))[kat]
    satirlar = puan.deneme(X[tr], y[tr], X[te], y[te], k)
    if k == KLER[0]:
        satirlar.append(puan.tam_lr(X[tr], y[tr], X[te], y[te]))
    for s in satirlar:
        s.update(veri=ds, kat=kat, k=k)
    return satirlar


if __name__ == "__main__":
    isler = [(ds, kat, k) for ds in VERILER for k in KLER for kat in range(5)]
    sonuc = Parallel(n_jobs=-1, verbose=5)(delayed(is_)(*a) for a in isler)
    os.makedirs("sonuclar", exist_ok=True)
    pd.DataFrame([s for b in sonuc for s in b]).to_csv("sonuclar/kiyaslama.csv", index=False)
    print("bitti")
