"""Chang, Tan, Lengerich, Goldenberg ve Caruana (2021) yeniden üretimi için GAM'ler.

Her model aynı arayüzü sunar:
    fit(X, y)            -> model
    olasilik(X)          -> P(y=1)
    katkilar(X)          -> (n, p) matris: her özniteliğin şekil fonksiyonunun o gözlemdeki değeri
Böylece farklı GAM'lerin şekil fonksiyonları birbiriyle ve "gerçek" ile karşılaştırılabilir.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _surekli_mi(x):
    return len(np.unique(x)) > 10


class EBM:
    """Explainable Boosting Machine (ağaç tabanlı, yoğun): her ağaç tek bir öznitelik kullanır."""
    ad = "EBM"

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingClassifier
        self.m = ExplainableBoostingClassifier(interactions=0, random_state=0, n_jobs=1)
        self.m.fit(X, y)
        return self

    def olasilik(self, X):
        return self.m.predict_proba(X)[:, 1]

    def katkilar(self, X):
        return np.asarray(self.m.eval_terms(X))


class XGBGAM:
    """Derinliği 1 olan XGBoost: her ağaç tek bölmeli bir kütük, yani toplamsal model."""
    ad = "XGB-GAM"

    def fit(self, X, y):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(max_depth=1, n_estimators=600, learning_rate=0.05, subsample=0.8,
                                   reg_lambda=1.0, n_jobs=1, random_state=0, verbosity=0)
        self.m.fit(X, y)
        return self

    def olasilik(self, X):
        return self.m.predict_proba(X)[:, 1]

    def katkilar(self, X):
        import xgboost as xgb
        return self.m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


class Spline:
    """Kübik spline GAM (pygam): sürekli özniteliklere spline, ikililere doğrusal terim."""
    ad = "Spline"

    def __init__(self, n_splines=20):
        self.n_splines = n_splines

    def fit(self, X, y):
        from pygam import LogisticGAM, l, s
        terimler = None
        for j in range(X.shape[1]):
            t = s(j, n_splines=self.n_splines) if _surekli_mi(X[:, j]) else l(j)
            terimler = t if terimler is None else terimler + t
        self.m = LogisticGAM(terimler).gridsearch(X, y, lam=np.logspace(-2, 2, 5), progress=False)
        return self

    def olasilik(self, X):
        return self.m.predict_proba(X)

    def katkilar(self, X):
        return np.column_stack([self.m.partial_dependence(term=j, X=X) for j in range(X.shape[1])])


class LR:
    """L2 cezalı lojistik regresyon: her şekil fonksiyonu düz bir çizgi."""
    ad = "LR"

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.m = LogisticRegression(C=1.0, max_iter=5000).fit(self.sc.transform(X), y)
        return self

    def olasilik(self, X):
        return self.m.predict_proba(self.sc.transform(X))[:, 1]

    def katkilar(self, X):
        return self.sc.transform(X) * self.m.coef_[0]


class ILR:
    """Gösterge LR: her öznitelik en fazla 32 kantil kutusuna bölünür, kutular tek tek kodlanır."""
    ad = "iLR"

    def fit(self, X, y, kutu=32):
        # sürekli öznitelik: kantil sınırları; kesikli öznitelik: kendi değerleri
        self.tur, self.sinirlar = [], []
        for j in range(X.shape[1]):
            if _surekli_mi(X[:, j]):
                self.tur.append("surekli")
                self.sinirlar.append(np.unique(np.quantile(X[:, j], np.linspace(0, 1, kutu + 1)[1:-1])))
            else:
                self.tur.append("kesikli")
                self.sinirlar.append(np.unique(X[:, j]))
        Z, self.bloklar = self._kodla(X)
        self.m = LogisticRegression(C=1.0, max_iter=5000).fit(Z, y)
        return self

    def _kodla(self, X):
        parcalar, bloklar, bas = [], [], 0
        for j, s in enumerate(self.sinirlar):
            if self.tur[j] == "surekli":
                idx, genislik = np.searchsorted(s, X[:, j], side="right"), len(s) + 1
            else:
                idx, genislik = np.searchsorted(s, X[:, j], side="left"), len(s)
            O = np.zeros((len(X), genislik))
            O[np.arange(len(X)), np.clip(idx, 0, genislik - 1)] = 1
            parcalar.append(O)
            bloklar.append((bas, bas + genislik))
            bas += genislik
        return np.hstack(parcalar), bloklar

    def olasilik(self, X):
        return self.m.predict_proba(self._kodla(X)[0])[:, 1]

    def katkilar(self, X):
        Z, _ = self._kodla(X)
        w = self.m.coef_[0]
        return np.column_stack([Z[:, a:b] @ w[a:b] for a, b in self.bloklar])


MODELLER = [EBM, XGBGAM, Spline, LR, ILR]


def sadakat_hatasi(K_tahmin, K_gercek):
    """Her şekil fonksiyonu ortalaması sıfır olacak biçimde merkezlenir; ortalama mutlak farkların toplamı."""
    a = K_tahmin - K_tahmin.mean(0)
    b = K_gercek - K_gercek.mean(0)
    return float(np.abs(a - b).mean(0).sum())
