"""Bölüm 8 şekilleri: sonuclar/*.csv dosyalarından çizer."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

SIRA = ["EBM", "XGB-GAM", "Spline", "LR", "iLR"]
RENK = {"EBM": "#2a78d6", "XGB-GAM": "#eb6834", "Spline": "#1baf7a", "LR": "#eda100", "iLR": "#e87ba4",
        "Doğrusal": "#eda100", "Kutulu": "#e87ba4"}
MURE, IKINCIL, IZGARA = "#14242a", "#5b6170", "#e6e4df"
TR = lambda b: FuncFormatter(lambda v, _: f"{v:.{b}f}".replace(".", ","))
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": IKINCIL, "axes.labelcolor": MURE, "xtick.color": IKINCIL,
    "ytick.color": IKINCIL, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": IZGARA, "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": MURE,
})
VERI = ["Breast", "COMPAS", "Churn", "Adult"]


def sekil_dogruluk(d, yol):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, v in zip(axes, VERI):
        s = d[d.veri == v].groupby("yontem").auc.agg(["mean", "std"]).reindex(SIRA)
        y = np.arange(len(SIRA))[::-1]
        ax.errorbar(s["mean"], y, xerr=s["std"], fmt="none", ecolor=IZGARA, elinewidth=3, capsize=0)
        ax.scatter(s["mean"], y, c=[RENK[m] for m in SIRA], s=70, zorder=3, edgecolor="white", linewidth=1.5)
        ax.set_yticks(y, SIRA)
        ax.set_title(v, loc="left")
        ax.xaxis.set_major_formatter(TR(3))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("test AUC")
        lo, hi = s["mean"].min(), s["mean"].max()
        ax.set_xlim(lo - (hi - lo) * 0.6 - 0.004, hi + (hi - lo) * 0.6 + 0.004)
    fig.suptitle("Beş GAM'in test AUC'si (5 rastgele bölme, nokta = ortalama, çizgi = ±1 standart sapma)",
                 x=0.06, ha="left", fontsize=11, color=IKINCIL)
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


def sekil_sadakat(d, yol):
    d = d.copy()
    # Her satırda (veri × üretici) hatayı en iyi öğreniciye göre oranla: 1 = en sadık
    d["oran"] = d.hata / d.groupby(["veri", "uretici"]).hata.transform("min")
    ureticiler = ["EBM", "XGB-GAM", "Spline", "LR"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, v in zip(axes, VERI):
        t = d[d.veri == v].pivot_table(index="uretici", columns="yontem", values="oran").reindex(index=ureticiler, columns=SIRA)
        goster = np.log2(t.values.clip(1, 8))
        ax.imshow(goster, cmap="Blues_r", vmin=0, vmax=3, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                val = t.values[i, j]
                metin = "en iyi" if val < 1.0005 else f"×{val:.1f}".replace(".", ",")
                ax.text(j, i, metin, ha="center", va="center", fontsize=9.5,
                        color="white" if goster[i, j] < 1.2 else MURE, fontweight="bold" if val < 1.0005 else "normal")
                if ureticiler[i] == SIRA[j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=MURE, lw=2))
        ax.set_xticks(range(5), SIRA, rotation=0, fontsize=9.5)
        ax.set_yticks(range(4), [f"üretici: {u}" for u in ureticiler] if v == "Breast" else [""] * 4, fontsize=9.5)
        ax.grid(False)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(v, loc="left")
    fig.suptitle("Şekil fonksiyonu hatası, en sadık öğreniciye oranla (çerçeve: üretici ile öğrenici aynı algoritma)",
                 x=0.06, ha="left", fontsize=11, color=IKINCIL)
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


def _egriler(ax, e, kalin=None):
    for m in ["EBM", "XGB-GAM", "Spline", "Doğrusal", "Kutulu"]:
        ax.plot(e.sicaklik, e[m], color=RENK[m], lw=2.2 if m in (kalin or []) else 1.6, label=m,
                ls="-" if m != "Kutulu" else (0, (4, 2)))
    ax.set_xlabel("sıcaklık (°C)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}".replace(",", ".")))


def sekil_bisiklet(gercek, esik, skor, yol):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    _egriler(a1, gercek)
    a1.set_ylabel("sıcaklığın kiralamaya katkısı (bisiklet/gün)")
    a1.set_title("Gerçek veri: aynı soruya beş farklı cevap", loc="left")
    a1.legend(frameon=False, fontsize=9.5, loc="lower right")

    a2.axvspan(28, esik.sicaklik.max(), color="#fbe3d6", zorder=0)
    a2.plot(esik.sicaklik, esik["Gerçek"], color=MURE, lw=3, label="gerçek etki", zorder=5)
    _egriler(a2, esik, kalin=["EBM", "XGB-GAM", "Spline"])
    a2.set_title("Yarı-sentetik: 28 °C üstünde 1.500 bisikletlik ani düşüş", loc="left")
    s = skor[skor.deney == "esik"].set_index("yontem")
    a2.text(28.2, a2.get_ylim()[0] + 0.05 * np.ptp(a2.get_ylim()), "sıcak günler", fontsize=10, color="#a3361f")
    satir = " · ".join(f"{m} {s.loc[m, 'rmse_gercege_sicak']:.0f}" for m in ["EBM", "XGB-GAM", "Spline", "Kutulu", "Doğrusal"])
    a2.text(0.0, -0.22, "Sıcak günlerde gerçeğe RMSE: " + satir, transform=a2.transAxes, fontsize=10, color=IKINCIL)
    a2.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("sekiller", exist_ok=True)
    sekil_dogruluk(pd.read_csv("sonuclar/dogruluk.csv"), "sekiller/sekil1_dogruluk.png")
    sekil_sadakat(pd.read_csv("sonuclar/sadakat.csv"), "sekiller/sekil2_sadakat.png")
    sekil_bisiklet(pd.read_csv("sonuclar/bisiklet_gercek_egriler.csv"), pd.read_csv("sonuclar/bisiklet_esik_egriler.csv"),
                   pd.read_csv("sonuclar/bisiklet_skor.csv"), "sekiller/sekil3_bisiklet.png")
