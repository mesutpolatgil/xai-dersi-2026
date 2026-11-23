"""Karşılaştırma tabloları ve şekiller."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from veriler import KOK

MAVI, TURUNCU, GRI, ZEMIN = "#2a78d6", "#eb6834", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": ZEMIN, "axes.facecolor": ZEMIN, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True, "grid.color": "#e6e5e1"})


def oku(ad):
    return pd.read_csv(KOK / "sonuclar" / f"{ad}.csv")


def _ms(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"


def tablo_ana():
    a = oku("ana")
    return a.groupby(["yas", "lam"]).agg(M=("M", lambda s: f"{s.min()}-{s.max()}"), test=("dogruluk", _ms),
                                         COMPAS=("compas", _ms), kural=("kural", lambda s: dict(s.value_counts().sort_index())),
                                         kanit=("kanit", "mean"), sure_sn=("sure", "max")).round(1)


def tablo_listeler(yas="yas_tarama"):
    a = oku("ana").query("yas == @yas")
    return a.groupby("lam").liste.value_counts().rename("kat sayısı").reset_index()


def tablo_rakip():
    r, a = oku("rakip"), oku("ana").query("yas == 'yas_tarama' and lam == 0.005")
    s = {k: g.dogruluk for k, g in r.groupby("yontem")}
    s["CORELS (λ=0.005)"], s["COMPAS puanı"] = a.dogruluk, a.compas
    t = pd.DataFrame({k: dict(ortalama=v.mean(), std=v.std()) for k, v in s.items()}).T
    return t.sort_values("ortalama", ascending=False).round(3)


def tablo_irk():
    a = oku("ana").query("yas == 'yas_tarama' and lam == 0.005")
    return pd.DataFrame({m: {f"{g} {o}": a[f"{m}_{g}_{o}"].mean() for g in ("siyah", "beyaz") for o in ("TPR", "FPR")}
                         for m in ("CORELS", "COMPAS")}).round(3)


def bootstrap_ozet():
    b = oku("bootstrap")
    r, q = b[b.ornek < 0].liste.iloc[0], b[b.ornek >= 0]
    return {"aynı kural kümesi %": 100 * (q.liste == r).mean(), "farklı kural kümesi sayısı": q.liste.nunique(),
            "kural sayısı": dict(q.kural.value_counts().sort_index()), "kanıtlı %": 100 * q.kanit.mean()}


def tablo_kendi():
    k = oku("kendi")
    t = k.groupby("yontem").agg(dogruluk=("dogruluk", _ms), TPR=("TPR", "mean"), FPR=("FPR", "mean"),
                                kural=("kural", "mean"), kanit=("kanit", "mean"))
    return t.assign(_o=k.groupby("yontem").dogruluk.mean()).sort_values("_o", ascending=False).drop(columns="_o").round(3)


def _kaydet(fig, ad):
    (KOK / "sekiller").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(KOK / "sekiller" / ad, dpi=150)
    plt.close(fig)


def sekil1():
    t = tablo_rakip()[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(t))
    renk = [MAVI if "CORELS" in i else TURUNCU if "COMPAS" in i else GRI for i in t.index]
    ax.hlines(y, t.ortalama - t["std"], t.ortalama + t["std"], color=renk, lw=2)
    ax.scatter(t.ortalama, y, color=renk, s=60, zorder=3)
    ax.set_yticks(y, t.index)
    ax.grid(axis="x")
    ax.set_xlabel("Test doğruluğu (10 kat ortalaması ± std)")
    ax.set_title("Katlar arası fark, yöntemler arası farktan büyük", loc="left")
    _kaydet(fig, "sekil1_kiyaslama.png")


def sekil2():
    t = tablo_irk()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(t))[::-1]
    for m, r, k in (("CORELS", MAVI, .18), ("COMPAS", TURUNCU, -.18)):
        ax.barh(y + k, t[m], height=.32, color=r, label=m)
        for yy, v in zip(y + k, t[m]):
            ax.text(v + .01, yy, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, .85)
    ax.set_title("İki yöntemde de siyahların TPR ve FPR'si daha yüksek", loc="left")
    ax.legend(frameon=False, loc="lower right")
    _kaydet(fig, "sekil2_irk.png")


def sekil3():
    q = oku("bootstrap").query("ornek >= 0")
    s = (q.liste.str.split(" ; ").explode().value_counts().head(8) / len(q) * 100)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(s.index, s.values, color=MAVI, height=.6)
    for i, v in enumerate(s.values):
        ax.text(v + 1, i, f"%{v:.0f}", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("100 bootstrap listesinde görülme (%)")
    ax.set_title("Kurallar ne kadar kararlı?", loc="left")
    _kaydet(fig, "sekil3_kararlilik.png")


if __name__ == "__main__":
    sekil1()
    sekil2()
    sekil3()
