"""Bisiklet uzantısı: gerçek veride ve 'sıcak hava eşiği' eklenmiş yarı-sentetik veride sıcaklık eğrileri (~1 dk)."""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import bisiklet as b

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    os.makedirs("sonuclar", exist_ok=True)
    X, y = b.bisiklet_verisi()
    izgara = np.linspace(np.quantile(X[:, 0], 0.01), np.quantile(X[:, 0], 0.99), 120)
    ort = X.mean(0)

    def egri(m):
        Xg = np.tile(ort, (len(izgara), 1))
        Xg[:, 0] = izgara
        k = m.katkilar(Xg)[:, 0]
        return k - m.katkilar(X)[:, 0].mean()  # veri üzerindeki ortalamaya göre merkezle

    # 1) Gerçek veri
    egriler, skor = {"sicaklik": izgara}, []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for M in b.MODELLER:
        m = M().fit(Xtr, ytr)
        skor.append(dict(deney="gercek", yontem=M.ad, rmse_test=float(np.sqrt(np.mean((m.tahmin(Xte) - yte) ** 2)))))
        egriler[M.ad] = egri(M().fit(X, y))
    pd.DataFrame(egriler).to_csv("sonuclar/bisiklet_gercek_egriler.csv", index=False)

    # 2) Yarı-sentetik: 28 °C üstünde 1.500 bisikletlik ani düşüş
    ys, gercek, taban = b.sicak_esik_verisi(X, y)
    g_egri = egri(taban) - b.DUSUS * (izgara > b.ESIK)
    g_egri = g_egri - np.interp(X[:, 0], izgara, g_egri).mean()
    egriler2 = {"sicaklik": izgara, "Gerçek": g_egri}
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, ys, gercek, test_size=0.3, random_state=0)
    sicak = Xte[:, 0] > b.ESIK
    for M in b.MODELLER:
        m = M().fit(Xtr, ytr)
        t = m.tahmin(Xte)
        skor.append(dict(deney="esik", yontem=M.ad,
                         rmse_test=float(np.sqrt(np.mean((t - yte) ** 2))),
                         rmse_gercege=float(np.sqrt(np.mean((t - gte) ** 2))),
                         rmse_gercege_sicak=float(np.sqrt(np.mean((t[sicak] - gte[sicak]) ** 2)))))
        e = egri(M().fit(X, ys))
        egriler2[M.ad] = e - np.interp(X[:, 0], izgara, e).mean()
    pd.DataFrame(egriler2).to_csv("sonuclar/bisiklet_esik_egriler.csv", index=False)
    pd.DataFrame(skor).to_csv("sonuclar/bisiklet_skor.csv", index=False)
    print(pd.DataFrame(skor).round(0))
