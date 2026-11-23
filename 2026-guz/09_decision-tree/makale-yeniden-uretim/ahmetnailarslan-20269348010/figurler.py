"""Makale tabloları, karşılaştırma tabloları ve şekiller."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import agac
import kararsizlik
import veri697

KOK = Path(__file__).resolve().parent
MAVI, TURUNCU, GRI = "#2a78d6", "#eb6834", "#52514e"
ZEMIN = "#fcfcfb"
AGACLAR = kararsizlik.AGACLAR

# Makale Tablo 3, 4, 5 (student.pdf, s. 6): (görev, ders) -> kurulum -> NV, DT, RF
_T = {("Bin", "Mat"): [(91.9, 90.7, 91.2), (83.8, 83.1, 83.0), (67.1, 65.3, 70.5)],
      ("Bin", "Por"): [(89.7, 93.0, 92.6), (87.5, 88.4, 90.1), (84.6, 84.4, 85.0)],
      ("5L", "Mat"): [(78.5, 76.7, 72.4), (60.5, 57.5, 52.7), (32.9, 31.5, 33.5)],
      ("5L", "Por"): [(72.9, 76.1, 73.5), (58.7, 62.9, 55.3), (31.0, 32.8, 36.7)],
      ("Reg", "Mat"): [(2.01, 1.94, 1.75), (2.80, 2.67, 2.46), (4.59, 4.46, 3.90)],
      ("Reg", "Por"): [(1.32, 1.46, 1.32), (1.89, 1.78, 1.79), (3.23, 2.93, 2.67)]}
MAKALE = pd.DataFrame([(g, d, k, *v) for (g, d), s in _T.items() for k, v in zip("ABC", s)],
                      columns=["gorev", "ders", "kurulum", "NV", "DT", "RF"]
                      ).set_index(["gorev", "ders", "kurulum"])


def ozet():
    """20 tekrarın ortalaması ve t dağılımına göre %95 güven aralığı yarı genişliği."""
    d = pd.read_csv(KOK / "sonuclar" / "sonuclar.csv")
    g = d.groupby(["gorev", "ders", "kurulum", "yontem"]).deger
    n = g.size().iloc[0]
    from scipy.stats import t
    ort = g.mean().unstack().loc[MAKALE.index, ["NV", "DT", "RF"]]
    ga = (g.std() * t.ppf(.975, n - 1) / np.sqrt(n)).unstack().loc[MAKALE.index, ["NV", "DT", "RF"]]
    return ort, ga


def tablo(gorev):
    """Makale (M) ve yeniden üretim (YÜ) yan yana."""
    ort, ga = ozet()
    b = 2 if gorev == "Reg" else 1
    s = {}
    for m in ("NV", "DT", "RF"):
        s[f"{m} M"] = MAKALE.loc[gorev, m].map(lambda v: f"{v:.{b}f}")
        s[f"{m} YÜ"] = [f"{o:.{b}f} ±{g:.{b}f}" for o, g in zip(ort.loc[gorev, m], ga.loc[gorev, m])]
    return pd.DataFrame(s)


def kazanan(df):
    """Her satırda en iyi yöntem (PCC'de büyük, RMSE'de küçük iyi); eşitlikler birlikte yazılır."""
    c = []
    for (g, *_), r in df.iterrows():
        r = r.round(2 if g == "Reg" else 1)
        c.append("/".join(r.index[r == (r.min() if g == "Reg" else r.max())]))
    return pd.Series(c, index=df.index)


def sekil1():
    ort, _ = ozet()
    plt.rcParams.update({"figure.facecolor": ZEMIN, "axes.facecolor": ZEMIN, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    for ax, gr, b in ((axs[0], ["Bin", "5L"], "Doğru sınıflama oranı (PCC, %)"), (axs[1], ["Reg"], "RMSE")):
        ix = [i for i in MAKALE.index if i[0] in gr]
        y = np.arange(len(ix))[::-1]
        for m, renk, kay in (("DT", MAVI, .16), ("RF", TURUNCU, -.16)):
            ad = "Karar ağacı" if m == "DT" else "Rastgele orman"
            ax.scatter(MAKALE.loc[ix, m], y + kay, s=62, facecolors="none", edgecolors=renk, linewidths=2, label=f"{ad} (makale)")
            ax.scatter(ort.loc[ix, m], y + kay, s=24, color=renk, label=f"{ad} (yeniden üretim)")
        ax.scatter(MAKALE.loc[ix, "NV"], y, marker="|", s=120, color=GRI, label="NV (makale)")
        ax.set_yticks(y, [f"{g} · {d} · {k}" for g, d, k in ix])
        ax.set_xlabel(b)
        ax.grid(axis="x", color="#e6e5e1")
        ax.set_axisbelow(True)
    axs[0].set_title("Yeniden üretim makalenin değerleriyle örtüşüyor", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="lower center", fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(KOK / "sekiller" / "sekil1_karsilastirma.png", dpi=150)
    plt.close(fig)


def agac_kur(ders, gorev, kurulum, cp=0.01):
    X, y, adlar, kat = agac.hazirla(ders, gorev, kurulum)
    m = agac.Agac(gorev != "Reg", kat, cp=cp).fit(X, y)
    return m, adlar, {"Bin": ["Kaldı", "Geçti"], "5L": agac.DUZEYLER, "Reg": None}[gorev]


def _ciz(ax, m, adlar, sinif):
    x = [0]

    def yer(d, der):
        if d["bol"] is None:
            d["_x"], d["_y"] = x[0], -der
            x[0] += 1
        else:
            yer(d["sol"], der + 1)
            yer(d["sag"], der + 1)
            d["_x"], d["_y"] = (d["sol"]["_x"] + d["sag"]["_x"]) / 2, -der
    yer(m.kok, 0)

    def ciz(d):
        yaprak = d["bol"] is None
        if not yaprak:
            for c, e in ((d["sol"], "evet"), (d["sag"], "hayır")):
                ax.plot([d["_x"], c["_x"]], [d["_y"], c["_y"]], color="#b5b4ae", lw=1, zorder=1)
                ax.text((d["_x"] + c["_x"]) / 2, (d["_y"] + c["_y"]) / 2, e, fontsize=6.5, color=GRI,
                        ha="center", va="center", bbox=dict(fc=ZEMIN, ec="none", pad=.5))
                ciz(c)
        t = f"{m.etiket(d, sinif)}\nn={d['n']}" if yaprak else m.kosul(d, adlar)
        ax.text(d["_x"], d["_y"], t, ha="center", va="center", fontsize=7.5, zorder=3,
                bbox=dict(boxstyle="round,pad=.3", fc="#dfeaf8" if yaprak else "white", ec=MAVI if yaprak else GRI, lw=1))
    ciz(m.kok)
    ax.axis("off")
    ax.margins(.12)


def sekil2():
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5), facecolor=ZEMIN)
    for ax, (d, g, k) in zip(axs.flat, AGACLAR):
        _ciz(ax, *agac_kur(d, g, k))
        ax.set_title(f"{k}-{d}-{g}", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(KOK / "sekiller" / "sekil2_agaclar.png", dpi=140)
    plt.close(fig)


# ---------- Kendi verim (UCI 697) ve kararsızlık ----------
def ozet697(sutun="deger"):
    from scipy.stats import t
    d = pd.read_csv(KOK / "sonuclar" / "sonuclar_697.csv")
    g = d.groupby(["gorev", "kurulum", "yontem"])[sutun]
    n = g.size().iloc[0]
    m = ["NV", "DT", "RF"]
    return g.mean().unstack()[m], (g.std() * t.ppf(.975, n - 1) / np.sqrt(n)).unstack()[m]


def tablo697(sutun="deger"):
    ort, ga = ozet697(sutun)
    t = pd.DataFrame({m: [f"{o:.1f} ±{g:.1f}" for o, g in zip(ort[m], ga[m])] for m in ort}, index=ort.index)
    t["en iyi"] = ort.idxmax(axis=1)
    return t


def tablo_sizinti():
    """A/B: eski kurulum ve satırlar çıkarılmış hâli yan yana (20 tekrar ort. ± %95 GA)."""
    from scipy.stats import t
    d = pd.read_csv(KOK / "sonuclar" / "sizinti_697.csv")
    e = pd.read_csv(KOK / "sonuclar" / "sonuclar_697.csv")
    n_tum = len(veri697.oku())
    cog = {"Bin": 67.9, "3S": 49.9}
    satir = []
    for g in veri697.GOREVLER:
        for k in "ABC":
            for c in ("eski", "kayit", "sinav", "gecen"):
                q = e[(e.gorev == g) & (e.kurulum == k)] if c == "eski" else d[(d.gorev == g) & (d.kurulum == k) & (d.cikar == c)]
                if q.empty:
                    continue
                r = {"görev": g, "kurulum": k, "çıkarılan": c, "n": n_tum if c == "eski" else int(q.n.iloc[0]),
                     "çoğunluk": f"{cog[g] if c == 'eski' else q.cogunluk.iloc[0]:.1f}"}
                for m in ("NV", "DT", "RF"):
                    v = q[q.yontem == m].deger
                    r[m] = f"{v.mean():.1f} ±{v.std() * t.ppf(.975, len(v) - 1) / np.sqrt(len(v)):.1f}"
                satir.append(r)
    return pd.DataFrame(satir)


def iddialar697():
    """Makalenin dört iddiası kendi verimizde."""
    ort, _ = ozet697()
    s1 = sum(ort.loc[(g, "A"), m] > ort.loc[(g, "B"), m] > ort.loc[(g, "C"), m]
             for g in veri697.GOREVLER for m in ort)
    s2 = sum(ort.loc[(g, "A")].idxmax() == "NV" for g in veri697.GOREVLER)
    s3 = int((ort.idxmax(axis=1) != "NV").sum())
    kok, sayi = [], []
    for g in veri697.GOREVLER:
        for k in "AB":
            m, adlar, _ = veri697.agac_kur(g, k)
            kul = m.ozet(adlar)[1]
            kok.append(kul[0])
            sayi.append(len(kul))
    return pd.DataFrame([
        ("1: A > B > C", f"{s1}/6 (görev × yöntem)", "✔" if s1 == 6 else "kısmen"),
        ("2: A'da en iyisi NV", f"{s2}/2 görev", "✔" if s2 == 2 else "✘" if s2 == 0 else "kısmen"),
        ("3: Ağaç tabanlılar önde (NN/SVM yok)", f"{s3}/6 satırda DT ya da RF birinci", "✔" if s3 == 6 else "kısmen"),
        ("4: Kökte notlar, az öznitelik", f"A/B ağaçlarında kök: {sorted(set(kok))}; öznitelik sayısı {min(sayi)}–{max(sayi)}",
         "✔" if all("dönem" in k for k in kok) and max(sayi) <= 5 else "kısmen")],
        columns=["iddia", "bulgu", "sonuç"])


def sekil3():
    ort, _ = ozet697()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=ZEMIN)
    for ax, (g, b) in zip(axs, (("Bin", "Dropout / diğer"), ("3S", "3 sınıf"))):
        ax.set_facecolor(ZEMIN)
        y = np.arange(3)[::-1]
        for m, renk, ad, isaret in (("NV", GRI, "NV", "D"), ("DT", MAVI, "Karar ağacı", "o"), ("RF", TURUNCU, "Rastgele orman", "s")):
            ax.scatter(ort.loc[g, m].loc[list("ABC")], y, s=70, color=renk, marker=isaret, label=ad, zorder=3)
        ax.hlines(y, ort.loc[g].min(axis=1).loc[list("ABC")], ort.loc[g].max(axis=1).loc[list("ABC")], color="#d5d4ce", lw=2)
        ax.set_yticks(y, list("ABC"))
        ax.set_ylim(-.5, 2.5)
        ax.set_xlabel("Doğru sınıflama oranı (%)")
        ax.set_title(b, loc="left")
        ax.grid(axis="x", color="#e6e5e1")
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(KOK / "sekiller" / "sekil3_697.png", dpi=150)
    plt.close(fig)


def kararsizlik_verisi():
    return pd.read_csv(KOK / "sonuclar" / "kararsizlik.csv")


def sekil4():
    o = kararsizlik.ozet(kararsizlik_verisi())
    fig, ax = plt.subplots(figsize=(8.5, 3.9), facecolor=ZEMIN)
    ax.set_facecolor(ZEMIN)
    y = np.arange(len(o))[::-1]
    for (c, ad, renk), kay in zip((("kök %", "kök bölünmesi", MAVI), ("üçü birden %", "kök + 2. düzeyin ikisi", TURUNCU),
                                   ("aynı öznitelik kümesi %", "aynı öznitelik kümesi", GRI)), (.26, 0, -.26)):
        ax.barh(y + kay, o[c], height=.23, color=renk, label=ad)
        for yy, v in zip(y + kay, o[c]):
            ax.text(v + 1, yy, f"%{v:.0f}", va="center", fontsize=8, color="#0b0b0b")
    ax.set_yticks(y, o["ağaç"])
    ax.set_xlim(0, 108)
    ax.set_xlabel("100 bootstrap ağacında aynen görülme (%)")
    ax.set_title("Kök kararlı, ağacın geri kalanı değil", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.45, -.16), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(KOK / "sekiller" / "sekil4_kararsizlik.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    (KOK / "sekiller").mkdir(exist_ok=True)
    sekil1()
    sekil2()
    sekil3()
    sekil4()
