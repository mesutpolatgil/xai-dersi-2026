"""Hastie, Tibshirani & Tibshirani (2020) simülasyonunun Python ile yeniden üretimi.

Dört yöntem: en iyi alt küme (best subset), ileri adımsal seçim (forward stepwise),
lasso ve gevşetilmiş lasso (relaxed lasso). Her yöntem ayrı bir doğrulama kümesinde
ayarlanır; değerlendirme gerçek Σ ile hesaplanan risk ölçüleriyle yapılır.
"""
from itertools import combinations

import numpy as np
from sklearn.linear_model import lasso_path

# Makaledeki 10 SNR değeri: 0,05 ile 6 arasında log ölçekte eşit aralıklı
SNR_DEGERLERI = np.exp(np.linspace(np.log(0.05), np.log(6), 10))

AYARLAR = {
    "low": dict(n=100, p=10, s=5, kmax=10, nlam=50),
    "medium": dict(n=500, p=100, s=5, kmax=50, nlam=100),
}


# ---------------------------------------------------------------- veri üretimi
def katsayi(beta_tipi, p, s):
    b = np.zeros(p)
    if beta_tipi == 1:            # s tane 1, yaklaşık eşit aralıklı
        b[np.round(np.linspace(0, p - 1, s)).astype(int)] = 1.0
    elif beta_tipi == 2:          # ilk s tane 1
        b[:s] = 1.0
    elif beta_tipi == 3:          # ilk s tane 10'dan 0,5'e eşit aralıklı
        b[:s] = np.linspace(10, 0.5, s)
    elif beta_tipi == 5:          # ilk s tane 1, sonra 0,5^(i-s) ile sönümlenen
        b[:s] = 1.0
        b[s:] = 0.5 ** np.arange(1, p - s + 1)
    else:
        raise ValueError(beta_tipi)
    return b


def kovaryans(p, rho):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def veri_uret(n, p, s, rho, snr, beta_tipi, rng):
    Sigma = kovaryans(p, rho)
    beta = katsayi(beta_tipi, p, s)
    sigma2 = beta @ Sigma @ beta / snr          # σ² = β₀ᵀΣβ₀ / SNR
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, p)) @ L.T
    Xv = rng.standard_normal((n, p)) @ L.T     # doğrulama kümesi, aynı boyutta
    y = X @ beta + rng.standard_normal(n) * np.sqrt(sigma2)
    yv = Xv @ beta + rng.standard_normal(n) * np.sqrt(sigma2)
    return X, y, Xv, yv, beta, Sigma, sigma2


# ---------------------------------------------------------------- yardımcılar
def _ortala(X, y):
    xm, ym = X.mean(0), y.mean()
    return X - xm, y - ym, xm, ym


def _ek_kareler(X, y, S):
    """S kümesi üzerinde en küçük kareler; tam boyutlu katsayı vektörü döner."""
    b = np.zeros(X.shape[1])
    if len(S):
        b[list(S)] = np.linalg.lstsq(X[:, list(S)], y, rcond=None)[0]
    return b


def _dogrulama_hatasi(B, a, Xv, yv):
    """B: (adaylar, p) katsayılar, a: (adaylar,) kesişimler."""
    tahmin = Xv @ B.T + a
    return ((yv[:, None] - tahmin) ** 2).mean(0)


# ---------------------------------------------------------------- yöntemler
def en_iyi_alt_kume(X, y, kmax):
    """Her k için eğitim RSS'ini en aza indiren alt kümeyi bulur.

    p ≤ 15 iken tüm alt kümeler taranır (kesin çözüm). Daha büyük p'de tam tarama
    imkânsızdır (p=100 için 2¹⁰⁰ alt küme); makale Gurobi ile karma tam sayılı
    optimizasyon kullanır. Biz burada aynı problemi yaklaşık çözen açık kaynaklı
    abess (splicing) algoritmasını kullanıyoruz.
    """
    Xc, yc, xm, ym = _ortala(X, y)
    p = X.shape[1]
    yol = [np.zeros(p)]
    if p > 15:
        from abess.linear import LinearRegression
        for k in range(1, kmax + 1):
            m = LinearRegression(support_size=[k], fit_intercept=False, thread=1)
            m.fit(Xc, yc)
            yol.append(np.asarray(m.coef_, dtype=float))
        B = np.array(yol)
        return B, ym - B @ xm
    for k in range(1, kmax + 1):
        en_iyi, en_iyi_rss = None, np.inf
        for S in combinations(range(p), k):
            b = _ek_kareler(Xc, yc, S)
            rss = ((yc - Xc @ b) ** 2).sum()
            if rss < en_iyi_rss:
                en_iyi, en_iyi_rss = b, rss
        yol.append(en_iyi)
    B = np.array(yol)
    return B, ym - B @ xm


def ileri_adimsal(X, y, kmax):
    """Her adımda kalıntıyla dikleştirilmiş korelasyonu en büyük değişkeni ekler."""
    Xc, yc, xm, ym = _ortala(X, y)
    n, p = Xc.shape
    Q = Xc.copy()                 # seçilenlere dik hale getirilmiş sütunlar
    r = yc.copy()
    secilen, yol = [], [np.zeros(p)]
    for _ in range(min(kmax, p, n - 1)):
        normlar = np.linalg.norm(Q, axis=0)
        skor = np.abs(Q.T @ r) / np.where(normlar > 1e-10, normlar, np.inf)
        skor[secilen] = -1
        j = int(np.argmax(skor))
        secilen.append(j)
        q = Q[:, j] / np.linalg.norm(Q[:, j])
        r = r - q * (q @ r)
        Q = Q - np.outer(q, q @ Q)
        yol.append(_ek_kareler(Xc, yc, secilen))
    B = np.array(yol)
    return B, ym - B @ xm


def _lasso_yolu(Xc, yc, nlam):
    n = len(yc)
    lam_max = np.abs(Xc.T @ yc).max() / n
    lamlar = np.exp(np.linspace(np.log(lam_max), np.log(lam_max * 1e-4), nlam))
    _, coefs, _ = lasso_path(Xc, yc, alphas=lamlar, max_iter=5000, tol=1e-6)
    return coefs.T                # (nlam, p)


def lasso(X, y, nlam):
    Xc, yc, xm, ym = _ortala(X, y)
    B = _lasso_yolu(Xc, yc, nlam)
    return B, ym - B @ xm


def gevsetilmis_lasso(X, y, nlam, ngamma=10):
    """β(λ,γ) = γ·β_lasso(λ) + (1−γ)·β_LS(λ); β_LS lasso'nun aktif kümesinde EKK."""
    Xc, yc, xm, ym = _ortala(X, y)
    B_lasso = _lasso_yolu(Xc, yc, nlam)
    adaylar = []
    for b in B_lasso:
        A = np.flatnonzero(b)
        b_ls = _ek_kareler(Xc, yc, A) if len(A) < len(yc) else b
        for g in np.linspace(1, 0, ngamma):
            adaylar.append(g * b + (1 - g) * b_ls)
    B = np.array(adaylar)
    return B, ym - B @ xm


YONTEMLER = {
    "En iyi alt küme": lambda X, y, a: en_iyi_alt_kume(X, y, a["kmax"]),
    "İleri adımsal": lambda X, y, a: ileri_adimsal(X, y, a["kmax"]),
    "Lasso": lambda X, y, a: lasso(X, y, a["nlam"]),
    "Gevşetilmiş lasso": lambda X, y, a: gevsetilmis_lasso(X, y, a["nlam"]),
}


# ---------------------------------------------------------------- ölçüler
def olculer(b, beta, Sigma, sigma2):
    d = b - beta
    hata = d @ Sigma @ d
    sinyal = beta @ Sigma @ beta
    return {
        "RR": hata / sinyal,                              # göreli risk
        "RTE": (hata + sigma2) / sigma2,                  # göreli test hatası
        "PVE": 1 - (hata + sigma2) / (sinyal + sigma2),   # açıklanan varyans oranı
        "sifirdan_farkli": int(np.sum(np.abs(b) > 1e-10)),
        "dogru_secilen": int(np.sum((np.abs(b) > 1e-10) & (beta != 0))),
    }


def tek_deneme(ayar, rho, snr, beta_tipi, tohum, yontemler=None):
    a = AYARLAR[ayar]
    rng = np.random.default_rng(tohum)
    X, y, Xv, yv, beta, Sigma, sigma2 = veri_uret(a["n"], a["p"], a["s"], rho, snr, beta_tipi, rng)
    satirlar = []
    for ad in (yontemler or YONTEMLER):
        B, a0 = YONTEMLER[ad](X, y, a)
        en_iyi = int(np.argmin(_dogrulama_hatasi(B, a0, Xv, yv)))
        satir = dict(ayar=ayar, rho=rho, snr=snr, beta_tipi=beta_tipi, tohum=tohum, yontem=ad)
        satir.update(olculer(B[en_iyi], beta, Sigma, sigma2))
        satirlar.append(satir)
    return satirlar
