"""Şekil 1-2 için tek senaryoda (ρ=0,35, β tipi 2) 100 tekrar."""
import warnings
import pandas as pd
from joblib import Parallel, delayed
import secim
warnings.filterwarnings("ignore")
isler = [(a, snr, t) for a in ["low", "medium"] for snr in secim.SNR_DEGERLERI for t in range(100)]
sonuc = Parallel(n_jobs=-1)(delayed(secim.tek_deneme)(a, 0.35, snr, 2, 50000 + i) for i, (a, snr, t) in enumerate(isler))
pd.DataFrame([s for b in sonuc for s in b]).to_csv("sonuclar/odak_sonuclar.csv", index=False)
