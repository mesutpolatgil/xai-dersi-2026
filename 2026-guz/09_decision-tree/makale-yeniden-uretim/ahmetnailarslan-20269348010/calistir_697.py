"""Kendi verim (UCI 697) deneyi ve ağaç kararsızlığı -> sonuclar/"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import agac
import kararsizlik
import veri697

TEKRAR = 20
SONUC = Path(__file__).resolve().parent / "sonuclar"


def calistir(n_jobs=-1):
    SONUC.mkdir(exist_ok=True)
    veri697.oku()
    isler = [(g, k, t) for g in veri697.GOREVLER for k in agac.KURULUMLAR for t in range(TEKRAR)]
    c = Parallel(n_jobs=n_jobs, verbose=2)(delayed(veri697.tek_tekrar)(*i) for i in isler)
    pd.DataFrame([s for b in c for s in b]).to_csv(SONUC / "sonuclar_697.csv", index=False)
    isler = [(g, k, c, d, t) for g in veri697.GOREVLER
             for k, d in (("A", "2nd"), ("B", "1st")) for c in ("kayit", "sinav", "gecen") for t in range(TEKRAR)]
    isler += [(g, "C", "sinav", "2nd", t) for g in veri697.GOREVLER for t in range(TEKRAR)]
    c = Parallel(n_jobs=n_jobs, verbose=2)(delayed(veri697.sizinti_tekrar)(*i) for i in isler)
    pd.DataFrame([s for b in c for s in b]).to_csv(SONUC / "sizinti_697.csv", index=False)
    kararsizlik.hepsi().to_csv(SONUC / "kararsizlik.csv", index=False)


if __name__ == "__main__":
    calistir()
    print("bitti")
