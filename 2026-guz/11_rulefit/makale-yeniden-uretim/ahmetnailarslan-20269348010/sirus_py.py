"""SIRUS (Bénard ve ark., AISTATS 2021) algoritmasının regresyon için Python uygulaması.

Bir kural, koşulların sıralı demetidir. Koşul: (öznitelik, kesim, yön)
  yön 0: kutu <= kesim ("<"),  yön 1: kutu > kesim (">=").
Kesim, kantil SIRASIDIR; bu yüzden farklı katlardaki kurallar karşılaştırılabilir.
"""
from collections import Counter

import numpy as np
from scipy.optimize import nnls
from scipy.stats import binom
from sklearn.ensemble import RandomForestRegressor


# ---------- 0. Kantil kutuları ----------
class Kutulayici:
    def __init__(self, q=10):
        self.q = q

    def fit(self, X):
        self.kesimler, self.siralar = [], []
        for x in np.asarray(X, float).T:
            d = np.unique(x)
            if len(d) <= self.q:              # ayrık öznitelik: değerlerin kendisi
                k = d[1:]
                s = np.arange(1, len(d))
            else:                             # sürekli: q-kantiller
                kantil = np.quantile(x, np.arange(self.q) / self.q)
                k = np.unique(kantil)
                k = k[k > d[0]]
                # tekrarlayan kantiller atılsa da kesim GERÇEK kantil sırasını korur
                s = np.array([np.flatnonzero(kantil == v)[0] for v in k])
            self.kesimler.append(k)
            self.siralar.append(s)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return np.column_stack([np.searchsorted(k, X[:, j], side="right")
                                for j, k in enumerate(self.kesimler)])


# ---------- 1. Kural üretimi ----------
def standart(yol):
    """Koşul sırasını standartlaştır (A & B ile B & A aynı kuraldır).

    x<3 & x<5 gibi yollar burada BİRLEŞTİRİLMEZ: birleştirmek tek koşullu kuralların
    sıklığını şişirir. Fazlalık olanları son işlem (rank denetimi) eler."""
    return tuple(sorted(yol))


def agactan_kurallar(agac):
    t, kurallar, yigin = agac.tree_, set(), [(0, ())]
    sol, sag, oz, esik = (t.children_left.tolist(), t.children_right.tolist(),
                          t.feature.tolist(), t.threshold.tolist())
    while yigin:
        d, yol = yigin.pop()
        if yol:
            kurallar.add(standart(yol))     # iç düğümler de kural
        if sol[d] != -1:
            j, c = oz[d], int(np.floor(esik[d]))
            yigin.append((sol[d], yol + ((j, c, 0),)))
            yigin.append((sag[d], yol + ((j, c, 1),)))
    return kurallar


def epsilon(frek, M, kmax=25):
    """Ek Bölüm 7: aynı veride iki uyarlamanın beklenen uyuşmazlığı."""
    f = np.sort(np.array(list(frek.values())))[::-1]
    u = np.unique(f[:2 * kmax])[::-1]
    izgara = (u[:-1] + u[1:]) / 2
    e = []
    for p0 in izgara:
        z = binom.cdf(M * p0, M, f)
        e.append((z * (1 - z)).sum() / max((1 - z).sum(), 1e-12))
    return float(np.mean(e)) if e else 0.0


def kural_frekanslari(B, y, derinlik=2, adim=1000, alfa=0.05, azami=20000, agac=None,
                      tohum=0):
    """Sığ rastgele orman; her kuralın görüldüğü ağaç oranı p̂(P)."""
    p = B.shape[1]
    rf = RandomForestRegressor(
        n_estimators=0, max_depth=derinlik, max_features=min(p, max(2, p // 3)),
        min_samples_split=5, bootstrap=True, warm_start=True, random_state=tohum)
    say = Counter()
    while True:
        eski = rf.n_estimators
        rf.n_estimators += agac or adim
        rf.fit(B, y)
        for a in rf.estimators_[eski:]:
            say.update(agactan_kurallar(a))
        M = rf.n_estimators
        frek = {k: v / M for k, v in say.items()}
        if agac or M >= azami or epsilon(frek, M) < alfa:
            return frek, M


# ---------- 2-3. Kural seçimi ve son işlem ----------
def kural_tut(kural, B):
    m = np.ones(len(B), bool)
    for j, c, yon in kural:
        m &= (B[:, j] > c) if yon else (B[:, j] <= c)
    return m


def son_islem(sirali, azami=50, n_nokta=3000):
    """Kendinden sık kuralların doğrusal birleşimi olan kuralı at.

    Veriye değil kurallara bağlıdır: göstergeler, kesim uzayından çekilen
    rastgele noktalarda değerlendirilir (Gram-Schmidt ile rank denetimi)."""
    p = 1 + max((j for r in sirali for j, _, _ in r), default=0)
    ust = np.ones(p, int)
    for r in sirali:
        for j, c, _ in r:
            ust[j] = max(ust[j], c + 1)
    U = np.random.default_rng(0).integers(0, ust + 1, size=(n_nokta, p))
    taban, secilen = np.ones((1, n_nokta)) / np.sqrt(n_nokta), []
    for r in sirali:
        v = kural_tut(r, U).astype(float)
        n0 = np.linalg.norm(v)
        for _ in range(2):
            v = v - (taban @ v) @ taban
        n1 = np.linalg.norm(v)
        if n0 > 0 and n1 > 1e-8 * n0:
            taban = np.vstack([taban, v / n1])
            secilen.append(r)
            if len(secilen) == azami:
                break
    return secilen


def sirala(frek):
    """Sıklık azalan; eşitlikte kısa kural ve "<" yönü önce (Ek Bölüm 4)."""
    return sorted(frek, key=lambda r: (-frek[r], len(r), sum(k[2] for k in r), r))


# ---------- 4. Birleştirme: beta >= 0 kısıtlı ridge ----------
def _nn_ridge(G, b, lam):
    L = np.linalg.cholesky(G + (lam + 1e-10 * np.trace(G)) * np.eye(len(G)))
    return nnls(L.T, np.linalg.solve(L, b))[0]


def nn_ridge_cv(Z, y, kat=10, tohum=0):
    """(1/n)||y-b0-Zb||^2 + lam||b||^2, b>=0; lam çapraz doğrulamayla."""
    def hazir(Z, y):
        zm, ym = Z.mean(0), y.mean()
        Zc = Z - zm
        return zm, ym, Zc.T @ Zc / len(y), Zc.T @ (y - ym) / len(y)
    zm, ym, G, b = hazir(Z, y)
    izgara = np.trace(G) / len(G) * np.logspace(-4, 1, 16)
    f = np.random.default_rng(tohum).permutation(len(y)) % kat
    hata = np.zeros(len(izgara))
    for k in range(kat):
        zk, yk, Gk, bk = hazir(Z[f != k], y[f != k])
        for i, lam in enumerate(izgara):
            w = _nn_ridge(Gk, bk, lam)
            hata[i] += ((y[f == k] - yk - (Z[f == k] - zk) @ w) ** 2).sum()
    w = _nn_ridge(G, b, izgara[hata.argmin()])
    return ym - zm @ w, w


# ---------- Model ----------
class SIRUS:
    def __init__(self, p0=None, q=10, derinlik=2, agac=None, azami_kural=50, tohum=0):
        self.p0, self.q, self.derinlik = p0, q, derinlik
        self.agac, self.azami_kural, self.tohum = agac, azami_kural, tohum

    def orman(self, X, y):
        """Pahalı kısım: orman + son işlemden geçmiş, sıklığa göre sıralı kurallar."""
        y = np.asarray(y, float)
        self.kutu = Kutulayici(self.q).fit(X)
        B = self.kutu.transform(X)
        self.frek, self.M = kural_frekanslari(B, y, self.derinlik, agac=self.agac,
                                              tohum=self.tohum)
        self.tum = son_islem(sirala(self.frek)[:400], self.azami_kural)
        S = np.column_stack([kural_tut(r, B) for r in self.tum])
        gecerli = S.any(0) & ~S.all(0)
        self.tum = [r for r, g in zip(self.tum, gecerli) if g]
        self.S = S[:, gecerli]
        self.ic = np.array([y[s].mean() for s in self.S.T])    # kural içi ortalama
        self.dis = np.array([y[~s].mean() for s in self.S.T])  # DEĞİLSE ortalaması
        self.p = np.array([self.frek[r] for r in self.tum])
        self.y, self._bellek = y, {}
        return self

    def _k(self, p0):
        return int((self.p > p0).sum())

    def agirlik(self, k):
        if k not in self._bellek:
            if k == 0:
                self._bellek[k] = (self.y.mean(), np.zeros(0))
            elif k == 1:
                self._bellek[k] = (0.0, np.ones(1))
            else:
                Z = np.where(self.S[:, :k], self.ic[:k], self.dis[:k])
                self._bellek[k] = nn_ridge_cv(Z, self.y, tohum=self.tohum)
        return self._bellek[k]

    def tahmin_k(self, X, k):
        b0, w = self.agirlik(k)
        B = self.kutu.transform(X)
        if k == 0:
            return np.full(len(B), b0)
        S = np.column_stack([kural_tut(r, B) for r in self.tum[:k]])
        return b0 + np.where(S, self.ic[:k], self.dis[:k]) @ w

    def secili(self, k):
        w = self.agirlik(k)[1]
        return {self.kimlik(r) for r, a in zip(self.tum[:k], w) if a > 0}

    def kimlik(self, kural):
        """Katlar arası karşılaştırma için: kesim yerine kantil sırası."""
        return tuple(sorted((j, int(self.kutu.siralar[j][c]), yon) for j, c, yon in kural))

    def fit(self, X, y):
        self.orman(X, y)
        self.k = self._k(self.p0)
        return self

    def predict(self, X):
        return self.tahmin_k(X, self.k)

    @staticmethod
    def _sade(kural):
        """Yalnızca YAZDIRMA için: x<3 & x<5 -> x<3 (sıklık sayımını etkilemez)."""
        en = {}
        for j, c, yon in kural:
            en[j, yon] = c if (j, yon) not in en else (min if yon == 0 else max)(c, en[j, yon])
        return [(j, c, yon) for (j, yon), c in en.items()]

    def yaz(self, kural, adlar):
        return " & ".join(f"{adlar[j]} {'>=' if yon else '<'} {self.kutu.kesimler[j][c]:.4g}"
                          for j, c, yon in self._sade(kural))


# ---------- Ölçütler ve p0 seçimi ----------
def dice(a, b):
    return 1.0 if not a and not b else 2 * len(a & b) / (len(a) + len(b))


def ort_dice(kumeler):
    return float(np.mean([dice(kumeler[i], kumeler[j])
                          for i in range(len(kumeler)) for j in range(i)]))


def cv_katlari(n, tohum, kat=10):
    return np.random.default_rng(tohum).permutation(n) % kat


def p0_egrisi(modeller, testler, y, izgara):
    """Kat modellerinden her p0 için (hata, kararlılık, kural sayısı)."""
    sonuc, bellek = [], {}
    for p0 in izgara:
        ks = tuple(m._k(p0) for m in modeller)
        if ks not in bellek:
            sse = sum(((yt - m.tahmin_k(Xt, k)) ** 2).sum()
                      for m, (Xt, yt), k in zip(modeller, testler, ks))
            kum = [m.secili(k) for m, k in zip(modeller, ks)]
            bellek[ks] = (sse / ((y - y.mean()) ** 2).sum(), ort_dice(kum),
                          np.mean([len(s) for s in kum]))
        sonuc.append(bellek[ks])
    return np.array(sonuc)


def p0_sec(egri, izgara):
    """(hata=0, kararlılık=0.9) noktasına en yakın p0; 1-25 kural aralığında."""
    u = np.hypot(egri[:, 0], egri[:, 1] - 0.9)
    u[(egri[:, 2] < 1) | (egri[:, 2] > 25)] = np.inf
    return float(izgara[u.argmin()])
