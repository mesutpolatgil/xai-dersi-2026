"""Makale ile karşılaştırma tabloları, kural listeleri ve şekiller."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calistir import KOK, SONUC, TABLO2A, TABLO2B, TABLO3
from sirus_py import SIRUS
from veri_yukle import MAKALE, sayisal, yukle

MAVI, TURUNCU, ZEMIN = "#2a78d6", "#eb6834", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": ZEMIN, "axes.facecolor": ZEMIN, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "axes.axisbelow": True})


def karsilastirma():
    """Makale (M) ile yeniden üretim (YÜ) yan yana."""
    a = pd.read_csv(SONUC / "ana.csv").set_index(["veri", "yontem"])
    al = lambda m, s: [a[s].get((v, m), np.nan) for v in MAKALE]
    kar = pd.DataFrame({"RuleFit M": TABLO2B.RuleFit, "pre YÜ": al("pre", "kararlilik"),
                        "imodels YÜ": al("imodels", "kararlilik"),
                        "SIRUS M": TABLO2B.SIRUS, "SIRUS YÜ": al("SIRUS", "kararlilik"),
                         "SIRUS-eşli YÜ": al("SIRUS-eşli", "kararlilik")},
                       index=MAKALE)
    say = pd.DataFrame({"RuleFit M": TABLO2A.RuleFit, "pre YÜ": al("pre", "kural"),
                        "imodels YÜ": al("imodels", "kural"),
                        "SIRUS M": TABLO2A.SIRUS, "SIRUS YÜ": al("SIRUS", "kural"),
                         "SIRUS-eşli YÜ": al("SIRUS-eşli", "kural")}, index=MAKALE)
    hata = pd.DataFrame({"RF M": TABLO3.RF, "RF YÜ": al("RF", "hata"),
                         "CART M": TABLO3.CART, "CART YÜ": al("CART", "hata"),
                         "RuleFit M": TABLO3.RuleFit, "pre YÜ": al("pre", "hata"),
                         "imodels YÜ": al("imodels", "hata"),
                         "SIRUS M": TABLO3.SIRUS, "SIRUS YÜ": al("SIRUS", "hata"),
                         "SIRUS-eşli YÜ": al("SIRUS-eşli", "hata")}, index=MAKALE)
    return kar.round(2), say.round(1), hata.round(2)


def kural_listesi(ad, p0):
    X, y, kat = yukle(ad)
    Xn, _ = sayisal(X, y, X, kat)
    m = SIRUS(p0=p0).fit(Xn, y)
    b0, w = m.agirlik(m.k)
    g = pd.DataFrame([dict(sıklık=m.frek[r], kural="EĞER " + m.yaz(r, list(X.columns)),
                           İSE=i, DEĞİLSE=d, ağırlık=a)
                      for r, i, d, a in zip(m.tum[:m.k], m.ic, m.dis, w) if a > 0])
    return m, b0, g


def pre_tablosu():
    """pre: makale, kutulu (soldan kapalı / sağdan kapalı) ve kutulamasız kararlılık; kural sayısı."""
    a = pd.read_csv(SONUC / "ana.csv").query("yontem == 'pre'").set_index("veri")
    h = pd.read_csv(SONUC / "pre_ham.csv").set_index("veri")
    s = pd.read_csv(SONUC / "pre_sag.csv").set_index("veri")
    return pd.DataFrame({"makale": TABLO2B.RuleFit, "kutulu [a,b)": a.kararlilik, "kutulu (a,b]": s.kararlilik,
                         "kutulamasız": h.kararlilik, "kural makale": TABLO2A.RuleFit,
                         "kural (pre)": a.kural, "kural (kutulamasız çalıştırma)": h.kural}
                        ).reindex(MAKALE + ["Bike"]).round(2)


def zaman_tablosu():
    """Bike: ilk %80 eğitim, son %20 test."""
    z = pd.read_csv(SONUC / "bike_zaman.csv").groupby("yontem")
    r = pd.read_csv(SONUC / "ana.csv").query("veri == 'Bike'").set_index("yontem").hata
    t = pd.DataFrame({"hata (zaman sıralı)": z.hata.mean(), "min": z.hata.min(), "maks": z.hata.max(),
                      "RMSE": z.rmse.mean(), "kural/yaprak": z.kural.mean()})
    t["hata (rastgele kat)"] = r.reindex(t.index)
    return t.reindex(["SIRUS", "pre", "CART", "RF"]).round(2)


def ozone_tablosu():
    return pd.read_csv(SONUC / "ozone_iki_liste.csv").round(3)


def _kaydet(fig, ad):
    (KOK / "sekiller").mkdir(exist_ok=True)
    fig.savefig(KOK / "sekiller" / ad, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sekil1():
    kar, _, _ = karsilastirma()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    yy = np.arange(len(kar))[::-1]
    for ad, mk, yu, renk, kay in (("RuleFit", "RuleFit M", "pre YÜ", TURUNCU, .12),
                                  ("SIRUS", "SIRUS M", "SIRUS YÜ", MAVI, -.12)):
        ax.scatter(kar[mk], yy + kay, s=70, facecolors="none", edgecolors=renk, linewidths=2, label=f"{ad} (makale)")
        ax.scatter(kar[yu], yy + kay, s=70, color=renk, label=f"{ad} (yeniden üretim)")
        ax.hlines(yy + kay, kar[mk], kar[yu], color=renk, lw=2, alpha=.5)
    ax.set_yticks(yy, kar.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Kararlılık (Dice-Sørensen)")
    ax.grid(axis="y", visible=False)
    ax.set_title("SIRUS sekiz veri setinin hepsinde RuleFit'ten kararlı", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.14))
    _kaydet(fig, "sekil1_kararlilik.png")


def sekil2():
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, ad in zip(axs, ("Ozone", "Bike")):
        e = pd.read_csv(SONUC / f"pareto_{ad}.csv")
        e = e[(e.kural >= 1) & (e.kural <= 25)]
        ax.plot(e.hata, e.kararlilik, color=MAVI, lw=2)
        for n in (3, 11):
            i = (e.kural - n).abs().idxmin()
            u = np.hypot(e.hata[i], e.kararlilik[i] - .9)
            ax.scatter(e.hata[i], e.kararlilik[i], s=60, color=MAVI, edgecolor=ZEMIN, linewidth=2, zorder=3)
            ax.annotate(f"{n} kural\nuzaklık {u:.2f}", (e.hata[i], e.kararlilik[i]), xytext=(-12, 2),
                        textcoords="offset points", fontsize=9, ha="right")
        ax.scatter(0, .9, marker="*", s=120, color="#52514e")
        ax.annotate("ideal", (0, .9), xytext=(6, 5), textcoords="offset points", fontsize=9)
        ax.set_title(ad, loc="left")
        ax.set_xlabel("Açıklanamayan varyans")
        ax.set_xlim(-.02, .8)
        ax.set_ylim(.35, 1.02)
    axs[0].set_ylabel("Kararlılık")
    fig.tight_layout()
    _kaydet(fig, "sekil2_p0.png")


if __name__ == "__main__":
    sekil1()
    sekil2()
