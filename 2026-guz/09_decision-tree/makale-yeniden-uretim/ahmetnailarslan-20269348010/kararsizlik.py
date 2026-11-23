"""Ağaç kararsızlığı: Şekil 4'teki 4 ağaç, 100 bootstrap örneğinde yeniden kurulur."""
import numpy as np
import pandas as pd

import agac

AGACLAR = [("Por", "Bin", "A"), ("Por", "5L", "A"), ("Por", "5L", "B"), ("Por", "Reg", "B")]
TEKRAR = 100


def _ust(m, adlar):
    """Kök ve ikinci düzeydeki iki bölünme (yoksa 'yaprak')."""
    def f(d):
        return "yaprak" if d["bol"] is None else m.kosul(d, adlar)
    k = m.kok
    if k["bol"] is None:
        return "yaprak", "yaprak", "yaprak"
    return f(k), f(k["sol"]), f(k["sag"])


def tam_agac(ders, gorev, kurulum):
    X, y, adlar, kat = agac.hazirla(ders, gorev, kurulum)
    return X, y, adlar, kat, agac.Agac(gorev != "Reg", kat).fit(X, y)


def calistir(ders, gorev, kurulum, B=TEKRAR, tohum=0):
    X, y, adlar, kat, ref = tam_agac(ders, gorev, kurulum)
    rng = np.random.default_rng(tohum)
    ad, satir = f"{kurulum}-{ders}-{gorev}", []

    def ekle(b, m):
        yaprak, kul = m.ozet(adlar)
        kok, sol, sag = _ust(m, adlar)
        satir.append(dict(agac=ad, ornek=b, kok=kok, evet=sol, hayir=sag, yaprak=yaprak,
                          oznitelik=";".join(kul)))
    ekle(-1, ref)                                   # -1: tüm veriyle kurulan ağaç
    for b in range(B):
        i = rng.integers(0, len(y), len(y))
        ekle(b, agac.Agac(gorev != "Reg", kat).fit(X[i], y[i]))
    return satir


def hepsi():
    return pd.DataFrame([s for a in AGACLAR for s in calistir(*a)])


def ozet(d):
    """Ağaç başına: tüm veriyle kurulan ağacın üst bölünmeleri kaç bootstrap ağacında aynen var?"""
    s = []
    for ad, g in d.groupby("agac", sort=False):
        r, b = g[g.ornek == -1].iloc[0], g[g.ornek >= 0]
        ayni = lambda c: (b[c] == r[c])
        s.append({"ağaç": ad, "kök": f"{r.kok}", "kök %": 100 * ayni("kok").mean(),
                  "2. düzey evet": r.evet, "evet %": 100 * ayni("evet").mean(),
                  "2. düzey hayır": r.hayir, "hayır %": 100 * ayni("hayir").mean(),
                  "üçü birden %": 100 * (ayni("kok") & ayni("evet") & ayni("hayir")).mean(),
                  "kök özniteliği aynı %": 100 * (b.kok.str.split().str[0] == r.kok.split()[0]).mean(),
                  "yaprak (tam / bootstrap medyan, min–maks)":
                      f"{r.yaprak} / {b.yaprak.median():.0f}, {b.yaprak.min()}–{b.yaprak.max()}",
                  "aynı öznitelik kümesi %":
                      100 * b.oznitelik.map(lambda t: set(t.split(";")) == set(r.oznitelik.split(";"))).mean()})
    return pd.DataFrame(s)


def oznitelik_sikligi(d):
    """Her özniteliğin bootstrap ağaçlarında kullanılma yüzdesi."""
    b = d[d.ornek >= 0]
    return (b.assign(o=b.oznitelik.str.split(";")).explode("o").groupby(["agac", "o"]).size()
            .unstack(0).reindex(columns=d.agac.unique()).fillna(0)
            / b.groupby("agac").size().iloc[0] * 100)
