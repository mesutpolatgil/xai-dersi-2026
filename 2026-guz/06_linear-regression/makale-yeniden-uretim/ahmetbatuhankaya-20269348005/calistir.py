"""Tüm simülasyon ızgarasını çalıştırır ve sonuclar.csv dosyasına yazar."""
import sys
import warnings
from itertools import product

import pandas as pd
from joblib import Parallel, delayed

import secim

warnings.filterwarnings("ignore")
TEKRAR = 20
RHOLAR = [0.0, 0.35, 0.7]
BETA_TIPLERI = [1, 2, 3, 5]

isler = [
    (ayar, rho, snr, bt, tohum)
    for ayar in sys.argv[1:] or ["low", "medium"]
    for rho, snr, bt in product(RHOLAR, secim.SNR_DEGERLERI, BETA_TIPLERI)
    for tohum in range(TEKRAR)
]
sonuc = Parallel(n_jobs=-1, verbose=0)(
    delayed(secim.tek_deneme)(ayar, rho, snr, bt, 1000 * i + tohum)
    for i, (ayar, rho, snr, bt, tohum) in enumerate(isler)
)
df = pd.DataFrame([satir for blok in sonuc for satir in blok])
df.to_csv("sonuclar/sonuclar.csv", index=False)
print(len(df), "satır yazıldı")
