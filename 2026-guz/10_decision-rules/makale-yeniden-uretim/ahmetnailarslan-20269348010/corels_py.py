"""CORELS (Angelino ve ark., JMLR 2018): kanıtlı en iyi kural listesi, sıfırdan Python.

Amaç: R(d) = yanlış sınıflama oranı + lam * (kural sayısı).
Dal-sınır araması; kullanılan sınırlar:
  - hiyerarşik alt sınır + ileri bakış (önek hatası + lam; bir kural daha en az lam ekler),
  - öncül desteği ve doğru sınıflanan destek >= lam,
  - eşdeğer noktalar sınırı (aynı öznitelikli, farklı etiketli örnekler hiç ayrılamaz),
  - permütasyon (aynı örnekleri yakalayan öneklerden yalnızca en iyisi tutulur).
"""
import itertools
import time

import numpy as np


def oncul_kazi(G, adlar, gruplar, alt=0.005, azami=2):
    """Tek ve iki koşullu öncüller; desteği [alt, 1-alt] dışında kalanlar elenir (Ek E.1)."""
    n = G.shape[1]
    aday = [(i,) for i in range(n)]
    if azami > 1:
        aday += [(i, j) for i, j in itertools.combinations(range(n), 2) if gruplar[i] != gruplar[j]]
    return [k for k in aday if alt <= G[:, list(k)].all(1).mean() <= 1 - alt]


def coz(G, y, onculler, lam, sure=600, azami_onek=400_000):
    """Dönen: dict(liste=[(oncul, tahmin)], varsayilan, amac, kanit, gezilen, sure)."""
    t0, N = time.time(), len(y)
    _, ilk, ters = np.unique(np.packbits(G, axis=1), axis=0, return_index=True, return_inverse=True)
    ters = ters.ravel()
    C = len(ilk)                                   # benzersiz öznitelik hücreleri
    poz = np.bincount(ters, y, C)
    neg = np.bincount(ters, minlength=C) - poz
    A = np.array([G[ilk][:, list(k)].all(1) for k in onculler])
    esd = np.minimum(poz, neg)                     # eşdeğer noktalar: kaçınılmaz hata
    en, secim = min(poz.sum(), neg.sum()) / N, ()
    katman, gorulen, gezilen, kanit = [(0.0, (), np.zeros(C, bool))], {}, 0, True
    while katman and kanit:
        yeni = []
        for hata, onek, kap in katman:
            if time.time() - t0 > sure or len(gorulen) > azami_onek:   # kanıt tamamlanmadı
                kanit = False
                break
            b = ~kap
            p, n = A[:, b] @ poz[b], A[:, b] @ neg[b]
            tp, tn, te = poz[b].sum(), neg[b].sum(), esd[b].sum()
            e = A[:, b] @ esd[b]
            K = len(onek) + 1
            h2 = hata + np.minimum(p, n)
            alt = (h2 + te - e) / N + lam * K
            amac = (h2 + np.minimum(tp - p, tn - n)) / N + lam * K
            for r in np.flatnonzero((p + n >= lam * N) & (np.maximum(p, n) >= lam * N)):
                gezilen += 1
                kural = (int(r), int(p[r] >= n[r]))
                if amac[r] < en - 1e-12:
                    en, secim = amac[r], onek + (kural,)
                if alt[r] + lam >= en - 1e-12:
                    continue
                k2 = kap | A[r]
                a = k2.tobytes()
                if gorulen.get(a, 9) <= alt[r] + 1e-12:
                    continue
                gorulen[a] = alt[r]
                yeni.append((h2[r], onek + (kural,), k2))
        katman = yeni
    liste = [(onculler[r], q) for r, q in secim]
    kap = np.zeros(N, bool)
    for k, _ in liste:
        kap |= G[:, list(k)].all(1)
    vars_ = int(y[~kap].mean() >= 0.5) if (~kap).any() else 0
    return dict(liste=liste, varsayilan=vars_, amac=float(en), kanit=kanit, gezilen=gezilen,
                sure=time.time() - t0, hucre=C)


def tahmin(m, G):
    t = np.full(len(G), m["varsayilan"])
    kap = np.zeros(len(G), bool)
    for k, q in m["liste"]:
        s = G[:, list(k)].all(1) & ~kap
        t[s] = q
        kap |= s
    return t


def yaz(m, adlar, evet="evet", hayir="hayır"):
    s, ek = [], "EĞER"
    for k, q in m["liste"]:
        s.append(f"{ek} {' VE '.join(adlar[i] for i in k)} İSE {evet if q else hayir}")
        ek = "DEĞİLSE EĞER"
    return "\n".join(s + [f"DEĞİLSE {evet if m['varsayilan'] else hayir}"])


def kimlik(m, adlar):
    """Sıradan bağımsız kural kümesi."""
    return " ; ".join(sorted(" & ".join(adlar[i] for i in k) + ("→1" if q else "→0") for k, q in m["liste"]))
