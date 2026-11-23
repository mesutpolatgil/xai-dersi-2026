"""Cortez & Silva (2008): veri, rpart benzeri karar ağacı (CART), rastgele orman, NV taban çizgisi.

Karar ağacı kütüphane çağrısı değildir: kategorik özniteliklerde alt küme bölünmesi ve
cp = 0.01 maliyet-karmaşıklık budaması R'daki rpart varsayılanlarına göre burada yazılmıştır.
"""
import io
import urllib.request
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

KOK = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DERSLER = {"Mat": "mat", "Por": "por"}
GOREVLER = ["Bin", "5L", "Reg"]
KURULUMLAR = {"A": [], "B": ["G2"], "C": ["G2", "G1"]}     # çıkarılan notlar
AZAMI_ALT_KUME = 8            # daha çok düzeyli kategoriklerde sıralama yöntemi
DUZEYLER = ["I", "II", "III", "IV", "V"]


# ---------- Veri ----------
def oku(ders):
    yol = KOK / "veri" / f"student-{DERSLER[ders]}.csv"
    if not yol.exists():
        yol.parent.mkdir(exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
        if "student.zip" in z.namelist():
            z = zipfile.ZipFile(io.BytesIO(z.read("student.zip")))
        yol.write_bytes(z.read(yol.name))
    return pd.read_csv(yol, sep=";")


def cevir(not_, gorev):
    """Notu göreve çevir: geçti/kaldı (>=10), Tablo 2'deki 5 düzey ya da notun kendisi."""
    not_ = np.asarray(not_)
    if gorev == "Bin":
        return (not_ >= 10).astype(int)
    if gorev == "5L":
        return np.select([not_ >= 16, not_ >= 14, not_ >= 12, not_ >= 10], [0, 1, 2, 3], 4)
    return not_.astype(float)


def hazirla(ders, gorev, kurulum):
    """X (tam sayı kodlu), y, öznitelik adları, kategorik düzeyler {sütun: düzeyler}."""
    d = oku(ders)
    X = d.drop(columns=["G3"] + KURULUMLAR[kurulum])
    kat = {}
    for j, c in enumerate(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            k = pd.Categorical(X[c])
            X[c] = k.codes
            kat[j] = list(k.categories)
    return X.to_numpy(float), cevir(d.G3, gorev), list(X.columns), kat


# ---------- Karar ağacı (CART / rpart) ----------
class Agac:
    """minsplit=20, minbucket=7, maxdepth=30, cp=0.01 (rpart varsayılanları)."""

    def __init__(self, siniflama, kat=(), cp=0.01, minsplit=20, minbucket=7, azami_derinlik=30):
        self.siniflama, self.kat, self.cp = siniflama, dict(kat), cp
        self.minsplit, self.minbucket, self.azami = minsplit, minbucket, azami_derinlik

    def fit(self, X, y):
        if self.siniflama:
            self.siniflar = np.unique(y)
            Y = (y[:, None] == self.siniflar).astype(float)      # Gini = one-hot kareler toplamı
        else:
            Y = y[:, None].astype(float)
        self.kok = self._buyut(X, Y, 0)
        self._buda(self.kok, self.cp * self.kok["risk"])
        return self

    def _dugum(self, Y):
        ort = Y.mean(0)
        risk = len(Y) * (1 - ort.max()) if self.siniflama else ((Y - ort) ** 2).sum()
        return dict(n=len(Y), ort=ort, risk=risk, bol=None)

    def _buyut(self, X, Y, derinlik):
        d = self._dugum(Y)
        if len(Y) < self.minsplit or derinlik >= self.azami or d["risk"] <= 1e-12:
            return d
        en, bol = 1e-9, None
        n, top = len(Y), Y.sum(0)
        taban = (top ** 2).sum() / n
        for j in range(X.shape[1]):
            x = X[:, j]
            if j in self.kat and len(self.kat[j]) > 2:           # alt küme bölünmesi
                v = np.unique(x)
                if len(v) > AZAMI_ALT_KUME:                      # 2^(k-1) alt küme taranamaz
                    x = self._sirala(x, Y)
                    k, e = self._sirali_bolunme(x, Y, top, taban)
                    if k > en + 1e-9:
                        en, bol = k, (j, frozenset(np.unique(X[x < e, j])))
                    continue
                for r in range(1, len(v) // 2 + 1):
                    for alt in combinations(v, r):
                        m = np.isin(x, alt)
                        nl = m.sum()
                        if min(nl, n - nl) < self.minbucket:
                            continue
                        sl = Y[m].sum(0)
                        k = (sl ** 2).sum() / nl + ((top - sl) ** 2).sum() / (n - nl) - taban
                        if k > en + 1e-9:
                            en, bol = k, (j, frozenset(alt))
                continue
            k, e = self._sirali_bolunme(x, Y, top, taban)
            if k > en + 1e-9:
                en, bol = k, (j, e)
        if bol is None:
            return d
        m = self._sol(X, bol)
        d["bol"], d["kazanc"] = bol, en
        d["sol"], d["sag"] = self._buyut(X[m], Y[m], derinlik + 1), self._buyut(X[~m], Y[~m], derinlik + 1)
        return d

    def _sirali_bolunme(self, x, Y, top, taban):
        """Sıralı bir öznitelikte en iyi kesim: (kazanç, eşik)."""
        n = len(Y)
        o = np.argsort(x, kind="stable")
        xs, cs = x[o], np.cumsum(Y[o], 0)[:-1]
        nl = np.arange(1, n)
        uygun = (xs[1:] > xs[:-1]) & (nl >= self.minbucket) & (n - nl >= self.minbucket)
        if not uygun.any():
            return -np.inf, None
        k = (cs ** 2).sum(1) / nl + ((top - cs) ** 2).sum(1) / (n - nl) - taban
        k[~uygun] = -np.inf
        i = int(k.argmax())
        return k[i], (xs[i] + xs[i + 1]) / 2

    @staticmethod
    def _sirala(x, Y):
        """Çok düzeyli kategorik: düzeyleri sınıf oranlarının 1. temel bileşenine göre sırala.

        İkili sınıf ve regresyonda en iyi alt küme bu sıralamadadır (Breiman ve ark., 1984);
        ikiden çok sınıfta yaklaşıktır."""
        v, ters = np.unique(x, return_inverse=True)
        c = np.bincount(ters).astype(float)
        P = np.column_stack([np.bincount(ters, Y[:, k]) for k in range(Y.shape[1])]) / c[:, None]
        Z = P - np.average(P, axis=0, weights=c)
        w = np.linalg.eigh((Z * c[:, None]).T @ Z)[1][:, -1]
        sira = np.empty(len(v))
        sira[np.argsort(Z @ w, kind="stable")] = np.arange(len(v))
        return sira[ters]

    @staticmethod
    def _sol(X, bol):
        j, e = bol
        return np.isin(X[:, j], list(e)) if isinstance(e, frozenset) else X[:, j] < e

    def _buda(self, d, alfa):
        """R(T) + alfa*|T| en küçük olacak biçimde alttan üste buda; (risk, yaprak) döner."""
        if d["bol"] is None:
            return d["risk"], 1
        (r1, y1), (r2, y2) = self._buda(d["sol"], alfa), self._buda(d["sag"], alfa)
        if d["risk"] + alfa <= r1 + r2 + alfa * (y1 + y2) + 1e-9:
            d["bol"] = None
            return d["risk"], 1
        return r1 + r2, y1 + y2

    def predict(self, X):
        cikti = np.empty(len(X))
        yigin = [(self.kok, np.arange(len(X)))]
        while yigin:
            d, i = yigin.pop()
            if d["bol"] is None:
                cikti[i] = self.siniflar[d["ort"].argmax()] if self.siniflama else d["ort"][0]
            else:
                m = self._sol(X[i], d["bol"])
                yigin += [(d["sol"], i[m]), (d["sag"], i[~m])]
        return cikti

    # -- okuma --
    def kosul(self, d, adlar):
        j, e = d["bol"]
        if isinstance(e, frozenset) and len(e) > 6:
            return f"{adlar[j]} ∈ {{{len(e)} düzey}}"
        if isinstance(e, frozenset):
            return f"{adlar[j]} ∈ {{{', '.join(self.kat[j][int(k)] for k in sorted(e))}}}"
        if j in self.kat:
            return f"{adlar[j]} = {self.kat[j][0]}"
        return f"{adlar[j]} < {e:g}"

    def etiket(self, d, sinif_adlari=None):
        if self.siniflama:
            s = self.siniflar[d["ort"].argmax()]
            return sinif_adlari[s] if sinif_adlari else str(s)
        return f"{d['ort'][0]:.1f}"

    def yaz(self, adlar, sinif_adlari=None):
        satir = []

        def gez(d, g, onek):
            if d["bol"] is None:
                satir.append(f"{'   ' * g}{onek}→ {self.etiket(d, sinif_adlari)}  (n={d['n']})")
                return
            satir.append(f"{'   ' * g}{onek}{self.kosul(d, adlar)} ?")
            gez(d["sol"], g + 1, "evet: ")
            gez(d["sag"], g + 1, "hayır: ")
        gez(self.kok, 0, "")
        return "\n".join(satir)

    def ozet(self, adlar):
        """Yaprak sayısı ve kullanılan öznitelikler (kökten itibaren)."""
        kul, yaprak, yigin = [], 0, [self.kok]
        while yigin:
            d = yigin.pop(0)
            if d["bol"] is None:
                yaprak += 1
                continue
            kul.append(adlar[d["bol"][0]])
            yigin += [d["sol"], d["sag"]]
        return yaprak, list(dict.fromkeys(kul))


# ---------- Rastgele orman ve taban çizgisi ----------
def orman(siniflama, p, tohum):
    """randomForest varsayılanları: T=500, mtry = √p (sınıflama) ya da p/3 (regresyon)."""
    if siniflama:
        return RandomForestClassifier(500, max_features=int(np.sqrt(p)), random_state=tohum)
    return RandomForestRegressor(500, max_features=max(1, p // 3), min_samples_split=5,
                                 random_state=tohum)


def nv(ders, gorev, kurulum, ytr):
    """Naif tahminci: A -> G2, B -> G1, C -> en sık sınıf / ortalama."""
    d = oku(ders)
    if kurulum != "C":
        return cevir(d["G2" if kurulum == "A" else "G1"], gorev)
    v, c = np.unique(ytr, return_counts=True)
    return np.full(len(d), v[c.argmax()] if gorev != "Reg" else ytr.mean())


def olcu(y, t, gorev):
    """Sınıflamada PCC (%), regresyonda RMSE."""
    return np.sqrt(np.mean((y - t) ** 2)) if gorev == "Reg" else 100 * np.mean(y == t)


def tek_tekrar(ders, gorev, kurulum, tohum):
    X, y, adlar, kat = hazirla(ders, gorev, kurulum)
    sin = gorev != "Reg"
    f = np.random.default_rng(tohum).permutation(len(y)) % 10
    T = {m: np.empty(len(y)) for m in ("NV", "DT", "RF")}
    for k in range(10):
        tr, te = f != k, f == k
        T["NV"][te] = nv(ders, gorev, kurulum, y[tr])[te]
        T["DT"][te] = Agac(sin, kat).fit(X[tr], y[tr]).predict(X[te])
        T["RF"][te] = orman(sin, X.shape[1], tohum).fit(X[tr], y[tr]).predict(X[te])
    return [dict(ders=ders, gorev=gorev, kurulum=kurulum, tekrar=tohum, yontem=m,
                 deger=olcu(y, t, gorev)) for m, t in T.items()]
