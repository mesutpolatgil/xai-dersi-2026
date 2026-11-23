"""Tablo 3-5: 2 ders × 3 görev × 3 kurulum, 20 tekrar × 10 kat -> sonuclar/sonuclar.csv"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import agac

TEKRAR = 20
SONUC = Path(__file__).resolve().parent / "sonuclar"


def calistir(n_jobs=-1):
    isler = [(d, g, k, t) for d in agac.DERSLER for g in agac.GOREVLER
             for k in agac.KURULUMLAR for t in range(TEKRAR)]
    for d in agac.DERSLER:
        agac.oku(d)          # veri bir kez insin
    c = Parallel(n_jobs=n_jobs, verbose=2)(delayed(agac.tek_tekrar)(*i) for i in isler)
    SONUC.mkdir(exist_ok=True)
    pd.DataFrame([s for b in c for s in b]).to_csv(SONUC / "sonuclar.csv", index=False)


if __name__ == "__main__":
    calistir()
    print("bitti")
