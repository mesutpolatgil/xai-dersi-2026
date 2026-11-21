"""Yeniden üretim şekilleri. sonuclar.csv ve gurultu_*.csv dosyalarından çizer."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

VIRGUL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
TR = {"season_SPRING": "mevsim: ilkbahar", "season_SUMMER": "mevsim: yaz", "season_FALL": "mevsim: sonbahar",
      "mevsim:SPRING": "mevsim: ilkbahar", "mevsim:SUMMER": "mevsim: yaz", "mevsim:FALL": "mevsim: sonbahar",
      "hava:MISTY": "hava: sisli", "hava:BAD": "hava: kötü"}

YONTEMLER = ["En iyi alt küme", "İleri adımsal", "Lasso", "Gevşetilmiş lasso"]
RENK = dict(zip(YONTEMLER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
CIZGI = {"En iyi alt küme": "-", "İleri adımsal": (0, (4, 2)), "Lasso": "-", "Gevşetilmiş lasso": "-"}
MURE, IKINCIL, IZGARA = "#1f2430", "#5b6170", "#e6e4df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": IKINCIL, "axes.labelcolor": MURE, "xtick.color": IKINCIL,
    "ytick.color": IKINCIL, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": IZGARA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": MURE,
})


def _uc_etiket(ax, x, ys, renkler, adlar, bosluk):
    """Çizgi sonlarına çakışmayan doğrudan etiketler."""
    sira = np.argsort(ys)
    konum = np.array(ys, float)[sira]
    for i in range(1, len(konum)):
        konum[i] = max(konum[i], konum[i - 1] + bosluk)
    for k, j in enumerate(sira):
        ax.annotate(adlar[j], (x, ys[j]), xytext=(x * 1.12, konum[k]), textcoords="data",
                    va="center", fontsize=10, color=MURE,
                    arrowprops=dict(arrowstyle="-", color=renkler[j], lw=1))


def _cizgiler(ax, tablo, deger, etiket=True, bosluk=0.004):
    for y in YONTEMLER:
        ax.plot(tablo.index, tablo[y], color=RENK[y], ls=CIZGI[y], lw=2,
                marker="o", ms=4, label=y)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 6])
    ax.set_xticklabels(["0,05", "0,1", "0,25", "0,5", "1", "2", "6"])
    ax.set_xlabel("Sinyal-gürültü oranı (SNR, log ölçek)")
    ax.set_ylabel(deger)
    ax.yaxis.set_major_formatter(VIRGUL)
    if etiket:
        x = tablo.index.max()
        _uc_etiket(ax, x, [tablo[y].iloc[-1] for y in YONTEMLER], [RENK[y] for y in YONTEMLER], YONTEMLER, bosluk)
        ax.set_xlim(right=x * 4.2)


def sekil_rte(d, yol):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ayar, baslik in zip(axes, ["low", "medium"], ["Düşük boyut: n=100, p=10, s=5", "Orta boyut: n=500, p=100, s=5"]):
        s = d[d.ayar == ayar]
        t = s.pivot_table(index="snr", columns="yontem", values="RTE", aggfunc="mean")
        _cizgiler(ax, t, "Göreli test hatası (RTE)", bosluk=(t.values.max() - t.values.min()) * 0.09)
        ax.set_title(baslik, loc="left")
    fig.suptitle("Göreli test hatası (düşük daha iyi) · ρ = 0,35, β tipi 2, 100 tekrar ortalaması",
                 x=0.06, ha="left", fontsize=11, color=IKINCIL)
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


def sekil_sifir(d, yol):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ayar, baslik in zip(axes, ["low", "medium"], ["Düşük boyut (p=10)", "Orta boyut (p=100)"]):
        s = d[d.ayar == ayar]
        t = s.pivot_table(index="snr", columns="yontem", values="sifirdan_farkli", aggfunc="mean")
        ax.axhline(5, color=IKINCIL, lw=1, ls=":")
        ax.text(0.052, 5.25, "gerçek sayı: 5", color=IKINCIL, fontsize=10) if ayar == "low" else ax.text(0.7, 3.2, "gerçek sayı: 5 (noktalı çizgi)", color=IKINCIL, fontsize=10)
        _cizgiler(ax, t, "Seçilen değişken sayısı", bosluk=(t.values.max() - t.values.min()) * 0.08)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(baslik, loc="left")
    fig.suptitle("Seçilen (sıfırdan farklı) katsayı sayısı · ρ = 0,35, β tipi 2",
                 x=0.06, ha="left", fontsize=11, color=IKINCIL)
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


def sekil_kazanan(d, yol):
    g = d.groupby(["ayar", "rho", "beta_tipi", "snr", "yontem"]).RTE.mean().unstack()
    oran = (g["Lasso"] < g["En iyi alt küme"]).groupby(level=["ayar", "snr"]).mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ayar, renk, ad in [("low", "#2a78d6", "Düşük boyut (p=10)"), ("medium", "#eb6834", "Orta boyut (p=100)")]:
        ax.plot(oran.index, 100 * oran[ayar], color=renk, lw=2, marker="o", ms=5, label=ad,
                ls="-" if ayar == "low" else (0, (4, 2)))
    ax.axhline(50, color=IKINCIL, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 6])
    ax.set_xticklabels(["0,05", "0,1", "0,25", "0,5", "1", "2", "6"])
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Sinyal-gürültü oranı (SNR, log ölçek)")
    ax.set_ylabel("Lasso'nun kazandığı durumlar (%)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Lasso ile en iyi alt küme: kim kazanıyor?", loc="left")
    ax.text(0.99, 0.97, "12 senaryo (3 ρ × 4 β tipi), RTE'ye göre", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=IKINCIL)
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


def sekil_bisiklet(secimler, sonuclar, yol):
    sahte = [c for c in secimler.columns if c.startswith("sahte")]
    gercek = [c for c in secimler.columns if c not in sahte + ["tekrar", "yontem"]]
    ort = secimler.groupby("yontem")[gercek].mean().T[YONTEMLER].rename(index=TR)
    ort.loc["20 sahte öznitelik (ort.)"] = secimler.groupby("yontem")[sahte].mean().mean(axis=1)[YONTEMLER]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    im = ax.imshow(ort.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [y.replace(" ", "\n", 1) for y in YONTEMLER])
    ax.set_yticks(range(len(ort)), ort.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for i in range(ort.shape[0]):
        for j in range(4):
            v = ort.values[i, j]
            ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 else MURE)
    ax.axhline(len(ort) - 1.5, color="white", lw=3)
    ax.set_title("Her özniteliğin seçilme sıklığı (%)", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)

    sahte_say = secimler.assign(n=secimler[sahte].sum(axis=1)).groupby("yontem").n.mean()[YONTEMLER]
    rmse = sonuclar.groupby("yontem").RMSE.mean()[YONTEMLER]
    y = np.arange(4)[::-1]
    ax2.barh(y, sahte_say.values, color=[RENK[k] for k in YONTEMLER], height=0.55)
    for yi, k in zip(y, YONTEMLER):
        ax2.text(sahte_say[k] + 0.2, yi, f"{sahte_say[k]:.1f}".replace(".", ",") + f" sahte · test RMSE {rmse[k]:.0f}",
                 va="center", fontsize=10, color=MURE)
    ax2.set_yticks(y, YONTEMLER)
    ax2.set_xlim(0, 14)
    ax2.set_xlabel("Seçilen sahte öznitelik sayısı (100 bölme ortalaması)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Tahmin aynı, açıklama farklı", loc="left")
    fig.tight_layout()
    fig.savefig(yol)
    plt.close(fig)


if __name__ == "__main__":
    d = pd.read_csv("sonuclar/sonuclar.csv")
    odak = pd.read_csv("sonuclar/odak_sonuclar.csv")
    sekil_rte(odak, "sekiller/sekil1_rte.png")
    sekil_sifir(odak, "sekiller/sekil2_secilen_sayi.png")
    sekil_kazanan(d, "sekiller/sekil3_kazanan.png")
    sekil_bisiklet(pd.read_csv("sonuclar/gurultu_secimler.csv"), pd.read_csv("sonuclar/gurultu_sonuclar.csv"), "sekiller/sekil4_bisiklet.png")
