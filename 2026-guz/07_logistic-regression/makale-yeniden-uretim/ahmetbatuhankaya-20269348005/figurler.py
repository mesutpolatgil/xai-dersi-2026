"""Bölüm 7 şekilleri: sonuclar/*.csv dosyalarından çizer."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

YONTEMLER = ["FasterRisk", "Lasso + ölçekle-yuvarla", "Lasso + yuvarlama", "Lasso + EKK (sürekli)"]
RENK = dict(zip(YONTEMLER, ["#0d7560", "#eb6834", "#2a78d6", "#8a63d2"]))
CIZGI = {"FasterRisk": "-", "Lasso + ölçekle-yuvarla": "-", "Lasso + yuvarlama": "-", "Lasso + EKK (sürekli)": (0, (2, 2))}
MURE, IKINCIL, IZGARA = "#14242a", "#5b6170", "#e6e4df"
VIRGUL = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": IKINCIL, "axes.labelcolor": MURE, "xtick.color": IKINCIL,
    "ytick.color": IKINCIL, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": IZGARA, "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": MURE,
})
BASLIK = {"mammo": "Mammo (n = 961)", "mushroom": "Mushroom (n = 8.124)", "adult": "Adult (n = 32.561)",
          "bank": "Bank (n = 41.188)"}


def _paneli_ciz(ax, t, tam, etiket):
    ks = t.index.values
    for y in YONTEMLER:
        ax.plot(ks, t[y], color=RENK[y], ls=CIZGI[y], lw=2.6 if y == "FasterRisk" else 1.8,
                marker="o", ms=5 if y == "FasterRisk" else 4, label=y, zorder=3 if y == "FasterRisk" else 2)
    ax.axhline(tam, color=MURE, lw=1.2, ls=(0, (6, 4)), label="Tam lojistik regresyon")
    ax.set_xticks(ks)
    alt, ust = ax.get_ylim()
    bas = 3 if ust - alt < 0.05 else 2
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, b=bas: f"{v:.{b}f}".replace(".", ",")))
    if etiket:
        ax.set_xlabel("puan kartındaki öznitelik sayısı (k)")


def sekil_kiyaslama(d, yol):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, ds in zip(axes, ["mammo", "mushroom", "adult", "bank"]):
        s = d[d.veri == ds]
        t = s[s.yontem != "Tam lojistik regresyon"].pivot_table(index="k", columns="yontem", values="auc_test")
        _paneli_ciz(ax, t, s[s.yontem == "Tam lojistik regresyon"].auc_test.mean(), True)
        ax.set_title(BASLIK[ds], loc="left")
    axes[0].set_ylabel("test AUC (5 katlı ÇD)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


def sekil_kredi(d, yol):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    t = d[d.yontem != "Tam lojistik regresyon"].pivot_table(index="k", columns="yontem", values="auc_test")
    _paneli_ciz(ax, t, d[d.yontem == "Tam lojistik regresyon"].auc_test.mean(), True)
    ax.set_ylabel("test AUC (5 katlı ÇD)")
    ax.set_title("German Credit: 1.000 başvuru, 25 ikili öznitelik", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("sekiller", exist_ok=True)
    sekil_kiyaslama(pd.read_csv("sonuclar/kiyaslama.csv"), "sekiller/sekil1_kiyaslama.png")
    sekil_kredi(pd.read_csv("sonuclar/kredi_cd.csv"), "sekiller/sekil2_kredi.png")
