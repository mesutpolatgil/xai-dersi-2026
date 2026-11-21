"""Gerçek veri uzantısı: kitabın 6. bölümündeki bisiklet kiralama verisinde dört yöntem."""
import warnings

import numpy as np
import pandas as pd
import pyreadr

import secim

warnings.filterwarnings("ignore")


def bisiklet_verisi(yol="../../ceviri/data/bike.RData"):
    d = pyreadr.read_r(yol)["bike"]
    # Kitabın 6. bölümündeki doğrusal modelin öznitelikleri
    X = pd.get_dummies(
        d[["season", "holiday", "workday", "weather", "temp", "hum", "windspeed", "days_since_2011"]],
        columns=["season", "holiday", "workday", "weather"], drop_first=True, dtype=float,
    )
    X.columns = [c.replace("season_", "mevsim:").replace("weather_", "hava:")
                 .replace("holiday_Y", "tatil").replace("workday_Y", "iş günü")
                 .replace("days_since_2011", "gün sayısı").replace("temp", "sıcaklık")
                 .replace("hum", "nem").replace("windspeed", "rüzgâr") for c in X.columns]
    return X, d["cnt"].to_numpy(float)


def bolme_deneyi(X, y, tekrar=100, tohum=0):
    """Rastgele %50 eğitim / %25 doğrulama / %25 test bölmeleri."""
    rng = np.random.default_rng(tohum)
    Xn = X.to_numpy()
    ayar = dict(kmax=Xn.shape[1], nlam=50)
    satirlar, secimler = [], []
    for t in range(tekrar):
        idx = rng.permutation(len(y))
        a, b = len(y) // 2, 3 * len(y) // 4
        tr, va, te = idx[:a], idx[a:b], idx[b:]
        # ölçekleme yalnızca eğitim kümesinden
        m, s = Xn[tr].mean(0), Xn[tr].std(0)
        Z = (Xn - m) / s
        for ad, f in secim.YONTEMLER.items():
            B, a0 = f(Z[tr], y[tr], ayar)
            k = int(np.argmin(secim._dogrulama_hatasi(B, a0, Z[va], y[va])))
            tahmin = Z[te] @ B[k] + a0[k]
            satirlar.append(dict(tekrar=t, yontem=ad,
                                 RMSE=np.sqrt(np.mean((y[te] - tahmin) ** 2)),
                                 sifirdan_farkli=int(np.sum(np.abs(B[k]) > 1e-10))))
            secimler.append(dict(tekrar=t, yontem=ad,
                                 **{c: float(abs(v) > 1e-10) for c, v in zip(X.columns, B[k])}))
    return pd.DataFrame(satirlar), pd.DataFrame(secimler)


def gurultu_ekle(X, k=20, tohum=1):
    """Hedefle hiçbir ilişkisi olmayan k tane sahte öznitelik ekler."""
    rng = np.random.default_rng(tohum)
    G = pd.DataFrame(rng.standard_normal((len(X), k)), columns=[f"sahte{i+1}" for i in range(k)], index=X.index)
    return pd.concat([X, G], axis=1)


if __name__ == "__main__":
    X, y = bisiklet_verisi()
    sonuc, secim_df = bolme_deneyi(X, y)
    sonuc.to_csv("bisiklet_sonuclar.csv", index=False)
    secim_df.to_csv("bisiklet_secimler.csv", index=False)
    print(sonuc.groupby("yontem")[["RMSE", "sifirdan_farkli"]].agg(["mean", "std"]).round(1))
    print(secim_df.groupby("yontem").mean(numeric_only=True).drop(columns="tekrar").T.round(2))
