"""Uzantı: kitabın 8. bölümündeki bisiklet verisinde aynı GAM sorusunu beş farklı GAM'e sormak.

1) Gerçek veri: cnt ~ sıcaklık + nem + rüzgâr + gün sayısı. Her GAM'in sıcaklık şekil fonksiyonu.
2) Yarı-sentetik "sıcak hava eşiği": gerçek sıcaklık etkisine 28 °C üstünde ani bir düşüş eklenir;
   hangi GAM bu sıçramayı geri bulabiliyor?
"""
import warnings

import numpy as np
import pandas as pd
import pyreadr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
OZNITELIK = ["temp", "hum", "windspeed", "days_since_2011"]
ADLAR = {"temp": "sıcaklık (°C)", "hum": "nem (%)", "windspeed": "rüzgâr (km/sa)", "days_since_2011": "gün sayısı"}
ESIK, DUSUS = 28.0, 1500.0


def bisiklet_verisi(yol="../../ceviri/data/bike.RData"):
    import os
    if not os.path.exists(yol) and os.path.exists("bike.RData"):
        yol = "bike.RData"
    d = pyreadr.read_r(yol)["bike"]
    return d[OZNITELIK].to_numpy(float), d["cnt"].to_numpy(float)


class _Model:
    def katkilar(self, X):
        raise NotImplementedError


class EBMReg(_Model):
    ad = "EBM"

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingRegressor
        self.m = ExplainableBoostingRegressor(interactions=0, random_state=0, n_jobs=1).fit(X, y)
        return self

    def tahmin(self, X):
        return self.m.predict(X)

    def katkilar(self, X):
        return np.asarray(self.m.eval_terms(X))


class XGBReg(_Model):
    ad = "XGB-GAM"

    def fit(self, X, y):
        import xgboost as xgb
        self.m = xgb.XGBRegressor(max_depth=1, n_estimators=600, learning_rate=0.05, subsample=0.8,
                                  n_jobs=1, random_state=0, verbosity=0).fit(X, y)
        return self

    def tahmin(self, X):
        return self.m.predict(X)

    def katkilar(self, X):
        import xgboost as xgb
        return self.m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


class SplineReg(_Model):
    ad = "Spline"

    def fit(self, X, y):
        from pygam import LinearGAM, s
        t = s(0, n_splines=20)
        for j in range(1, X.shape[1]):
            t = t + s(j, n_splines=20)
        self.m = LinearGAM(t).gridsearch(X, y, lam=np.logspace(-2, 3, 6), progress=False)
        return self

    def tahmin(self, X):
        return self.m.predict(X)

    def katkilar(self, X):
        return np.column_stack([self.m.partial_dependence(term=j, X=X) for j in range(X.shape[1])])


class LinReg(_Model):
    ad = "Doğrusal"

    def fit(self, X, y):
        self.m = LinearRegression().fit(X, y)
        return self

    def tahmin(self, X):
        return self.m.predict(X)

    def katkilar(self, X):
        return X * self.m.coef_


class BinReg(_Model):
    """Kutulu doğrusal model (iLR'nin regresyon karşılığı): 20 kantil kutusu, ridge."""
    ad = "Kutulu"

    def fit(self, X, y, kutu=20):
        self.s = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, kutu + 1)[1:-1])) for j in range(X.shape[1])]
        Z, self.b = self._kodla(X)
        self.m = Ridge(alpha=1.0).fit(Z, y)
        return self

    def _kodla(self, X):
        P, B, bas = [], [], 0
        for j, s in enumerate(self.s):
            idx, g = np.searchsorted(s, X[:, j], side="right"), len(s) + 1
            O = np.zeros((len(X), g))
            O[np.arange(len(X)), idx] = 1
            P.append(O)
            B.append((bas, bas + g))
            bas += g
        return np.hstack(P), B

    def tahmin(self, X):
        return self.m.predict(self._kodla(X)[0])

    def katkilar(self, X):
        Z, _ = self._kodla(X)
        return np.column_stack([Z[:, a:b] @ self.m.coef_[a:b] for a, b in self.b])


MODELLER = [EBMReg, XGBReg, SplineReg, LinReg, BinReg]


def sicak_esik_verisi(X, y, tohum=0):
    """Gerçek veriye spline GAM uydurulur; tahmine 28 °C üstünde 1.500 bisikletlik düşüş eklenir."""
    taban = SplineReg().fit(X, y)
    gercek = taban.tahmin(X) - DUSUS * (X[:, 0] > ESIK)
    artik = np.std(y - taban.tahmin(X))
    rng = np.random.default_rng(tohum)
    return gercek + rng.normal(0, artik, len(y)), gercek, taban


def sicaklik_egrileri(modeller, izgara):
    """Diğer öznitelikler ortalamada sabitken her modelin sıcaklık şekil fonksiyonu (merkezlenmiş)."""
    sonuc = {}
    for m in modeller:
        Xg = np.tile(m._ort, (len(izgara), 1))
        Xg[:, 0] = izgara
        k = m.katkilar(Xg)[:, 0]
        sonuc[m.ad] = k - k.mean()
    return sonuc
