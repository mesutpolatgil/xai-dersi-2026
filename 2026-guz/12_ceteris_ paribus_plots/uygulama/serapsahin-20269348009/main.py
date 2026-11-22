# %% [markdown]
# # Bölüm 12 – Ceteris Paribus Grafikleri: Uygulama
#
# **Öğrenci:** Serap Şahin (20269348009)
# **Kaynak bölüm:** Molnar, *Interpretable Machine Learning*, Bölüm 12 – Ceteris Paribus Plots
#
# Bu uygulama üç kısımdan oluşur:
#
# 1. **CP'yi sıfırdan yazmak.** Kitaptaki algoritma (sayısal ve kategorik öznitelik için) `numpy`/`pandas` ile kodlanır ve bağımsız bir referans uygulamayla doğrulanır.
# 2. **Kitabı yeniden üretmek.** Bölümdeki penguen ve bisiklet grafikleri (tek nokta CP, "ikili bağımlılıkların ötesine bak",
#    tüm öznitelikler için what-if, sparklines, model karşılaştırması) Python'da yeniden çizilir.
# 3. **Asıl uygulama: IBM Telco Customer Churn.** Tek bir müşteri için modelin ayrılma tahminini CP ile açıklıyoruz.
#    Ardından değişkenler arasındaki bağımlılıkları bozmanın tahminleri nasıl etkilediğini inceliyoruz.
#
# Tanım (kitaptaki gösterimle): $\mathbf{x}^{(i)}$ veri noktası ve $j$ özniteliği için CP eğrisi
#
# $$\left\{\left(z_k,\ \hat{f}\big(\mathbf{x}^{(i)}_{x_j := z_k}\big)\right)\right\}_{k=1}^{K}$$
#
# yani diğer her şey sabitken ("ceteris paribus") yalnızca $x_j$ değiştirildiğinde tahminin izlediği yoldur.

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# ---- Kitaptaki grafiklerin görünümü ----
# Kitap R'de iki görünüm kullanır: ggplot2 theme_bw (saçılım grafiği) ve DALEX/ceterisParibus'un "drwhy" teması (CP grafikleri).
# Varsayılan olarak theme_bw kurulur; CP grafikleri aşağıdaki drwhy() yardımcılarıyla biçimlenir.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 11,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8, "axes.labelsize": 13, "axes.titlesize": 12,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "axes.grid.which": "both", "grid.color": "#EBEBEB", "grid.linewidth": 0.7,
    "xtick.color": "#333333", "ytick.color": "#333333", "xtick.labelcolor": "#4D4D4D", "ytick.labelcolor": "#4D4D4D",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False,
})
for _anahtar in ("xtick.minor.ndivs", "ytick.minor.ndivs"):      # ggplot2: iki ana çizgi arasında tek ara çizgi
    if _anahtar in plt.rcParams:
        plt.rcParams[_anahtar] = 2

MOR, TURKUAZ, GRI_CIZGI = "#371EA3", "#46BAC2", "#B3B3B3"      # DALEX "drwhy" renkleri
VIRIDIS_3 = ["#440154", "#21908C", "#FDE725"]                   # ggplot2 scale_color_viridis_d(), 3 sınıf
GG_MODEL = {"lm": ("#F8766D", "-"), "randomForest": ("#7CAE00", (0, (2, 2))),   # ggplot2 varsayılan renk ve
            "svm": ("#00BFC4", (0, (4, 2))), "tree": ("#C77CFF", (0, (4, 4)))}  # çizgi tipleri (solid, 22, 42, 44)


def drwhy(ax, izgara="y"):
    """DALEX drwhy teması: çerçeve yok, yalnızca tek yönde açık gri ızgara, mor yazılar."""
    for kenar in ax.spines.values():
        kenar.set_visible(False)
    ax.grid(False, which="both")
    ax.grid(True, axis=izgara, which="both", color="#DEDEDE", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0, labelcolor=MOR)
    ax.xaxis.label.set_color(MOR)
    ax.yaxis.label.set_color(MOR)


def drwhy_baslik(ax, ozellik):
    """ceterisParibus panel başlığı: değişken adı sol üstte, alt çizgiler boşluk olarak."""
    ax.set_title(ozellik.replace("_", "  "), loc="left", color=MOR, fontsize=13)

URL = {
    "bike.csv": "https://raw.githubusercontent.com/christophM/interpretable-ml-book/8f470c85e5ed4ebd69de74425cb9832f1bdc25eb/data/bike.csv",
    "penguins.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/71e2436a092d714350de0fc409ca8a8714e7e78f/penguins.csv",
    "telco_churn.csv": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv",
}


def veri_oku(ad):
    """Veri setini GitHub'dan indirir (internet bağlantısı gerekir)."""
    return pd.read_csv(URL[ad])



# %% [markdown]
# ## 1. CP'yi sıfırdan yazmak
#
# Kitaptaki algoritma:
#
# * **Sayısal öznitelik:** eşit aralıklı ızgara $z_1,\dots,z_K$ (min'den max'a) kur; her $z_k$ için $x_j := z_k$ yapılmış kopyayı
#   oluştur ve tahmin al; çizgiyi çiz, gerçek değeri nokta ile işaretle.
# * **Kategorik öznitelik:** ızgara yerine benzersiz kategorileri kullan, çubuk grafik çiz.
#
# Tek bir fonksiyon iki durumu da çözer: ızgara sayılardan da kategorilerden de oluşabilir.

# %%
def cp_profili(tahmin_fn, x, ozellik, izgara):
    """x: tek satırlık DataFrame. Izgaradaki her değer için tahmini döndürür."""
    kopya = pd.concat([x] * len(izgara), ignore_index=True)
    kopya[ozellik] = list(izgara)
    return np.asarray(tahmin_fn(kopya))


def sayisal_izgara(sutun, K=60):
    return np.linspace(sutun.min(), sutun.max(), K)


def cp_ciz(ax, tahmin_fn, x, ozellik, izgara, renk=TURKUAZ, etiket=None, nokta=True, nokta_renk=None, **kw):
    """Sayısal CP eğrisi + gözlemin kendisi (nokta). Varsayılan renkler DALEX'inkilerdir."""
    y = cp_profili(tahmin_fn, x, ozellik, izgara)
    ax.plot(izgara, y, color=renk, lw=1.8, label=etiket, **kw)
    if nokta:
        ax.scatter(x[ozellik], tahmin_fn(x), color=nokta_renk or (MOR if renk == TURKUAZ else renk), s=70, zorder=5)
    return y


def cp_cubuk(ax, tahmin_fn, x, ozellik, kategoriler):
    """Kategorik CP, DALEX biçiminde: çubuklar gözlemin tahmininden başlar (kesikli çizgi), gerçek kategori en altta noktayla."""
    gercek = x[ozellik].iloc[0]
    sira = [gercek] + sorted(k for k in kategoriler if k != gercek)
    y = cp_profili(tahmin_fn, x, ozellik, sira)
    p = float(np.asarray(tahmin_fn(x))[0])
    konum = np.arange(len(sira))
    ax.barh(konum, y - p, left=p, height=0.75, color=TURKUAZ)
    ax.axvline(p, color=MOR, ls="--", lw=1)
    ax.scatter([p], [0], color=MOR, s=70, zorder=5)
    ax.set_yticks(konum, [str(k) for k in sira])
    drwhy(ax, izgara="x")
    drwhy_baslik(ax, ozellik)
    return dict(zip(sira, y))


# %% [markdown]
# ## 2. Kitabın örneklerini yeniden üretmek
#
# ### 2.1 Penguen: tek bir penguen için gaga derinliği CP'si
#
# Kitaptaki kurulum: 2/3 eğitim, rastgele orman, `sex ~ species + bill_depth + bill_length + flipper_length + body_mass`.
# Türkçe çeviride Şekil 12.2, dişi olasılığı sütunu açıkça seçilerek yeniden çizildi.
# Python uygulamasında da aynı sınıfın olasılığı, P(dişi), gösteriliyor.
#
# Kitabın şekli **gaga derinliği 20,6 mm olan erkek bir Adelie pengueni** açıklar (veride bu değere sahip tek penguen:
# Torgersen adası, gaga uzunluğu 39,3 mm). Aynı gözlemi açıklamak için bu pengueni **açıkça seçiyor** ve modelin onu eğitimde
# görmemesi için eğitim/test ayrımından önce veriden ayırıyorum.
#
# *Not:* R çevirisi ve Python uygulaması aynı gözlemi ve sınıfı açıklar. Model uygulaması ve eğitim ayrımı
# farklı olduğundan tahmin değerleri ile eğrinin ayrıntıları değişebilir.

# %%
peng = veri_oku("penguins.csv").dropna()
P_OZ = ["species", "bill_depth_mm", "bill_length_mm", "flipper_length_mm", "body_mass_g"]
kitap_maske = (peng["species"] == "Adelie") & (peng["bill_depth_mm"] == 20.6) & (peng["sex"] == "MALE")
assert kitap_maske.sum() == 1, "Kitaptaki penguen veride tek olmalı"
kitap_penguen = peng[kitap_maske]
p_egitim, p_test = train_test_split(peng[~kitap_maske], train_size=2 / 3, random_state=2)
p_test = pd.concat([kitap_penguen, p_test])                      # kitaptaki penguen test kümesinde, eğitimde değil
peng_rf = Pipeline([
    ("on", ColumnTransformer([("kat", OneHotEncoder(handle_unknown="ignore"), ["species"])], remainder="passthrough")),
    ("rf", RandomForestClassifier(n_estimators=500, random_state=2, n_jobs=-1)),
]).fit(p_egitim[P_OZ], p_egitim["sex"] == "FEMALE")
disi_sutunu = list(peng_rf.classes_).index(True)  # Eğitim hedefi sex == "FEMALE".


def p_disi(D):
    return peng_rf.predict_proba(D)[:, disi_sutunu]


penguen = kitap_penguen[P_OZ].reset_index(drop=True)
assert not kitap_maske.loc[p_egitim.index].any()
p_disi_baslangic = p_disi(penguen)[0]
print(f"İncelenen penguen: {penguen['species'][0]}, ada {kitap_penguen['island'].iloc[0]}, gaga derinliği "
      f"{penguen['bill_depth_mm'][0]} mm, gerçek etiket: {kitap_penguen['sex'].iloc[0]}, "
      f"P(dişi) = {p_disi_baslangic:.2f}, P(erkek) = {1 - p_disi_baslangic:.2f}")

# %% [markdown]
# ### Doğrulama: CP kodumuz doğru mu?
#
# `sklearn.inspection.partial_dependence`, tek gözlemle varsayılan ızgarada yalnızca penguenin gözlenen gaga
# derinliğini kullanır. Bu nedenle birden çok derinlikte CP hesabını **bağımsız bir referans döngüsüyle** doğruluyoruz.
# Döngü modeli her derinlik için ayrı çağırır. `cp_profili` tüm satırları tek seferde tahmin eder.

# %%
def cp_referans(tahmin_fn, x, ozellik, degerler):
    """Bağımsız referans: her değer için ayrı kopya, ayrı model çağrısı."""
    sonuc = []
    for z in degerler:
        kopya = x.copy()
        kopya[ozellik] = z
        sonuc.append(float(np.asarray(tahmin_fn(kopya))[0]))
    return np.array(sonuc)


kontrol_derinlik = [13.1, 15.0, 17.5, 18.4, 20.6, 21.5]
cp_hizli = cp_profili(p_disi, penguen, "bill_depth_mm", kontrol_derinlik)
cp_ref = cp_referans(p_disi, penguen, "bill_depth_mm", kontrol_derinlik)
print(pd.DataFrame({"gaga derinliği": kontrol_derinlik, "cp_profili": cp_hizli, "referans döngü": cp_ref}).round(4).to_string(index=False))
assert np.allclose(cp_hizli, cp_ref)
print("✔ Sayısal öznitelik: 6 farklı gaga derinliğinde cp_profili = bağımsız referans")

# %% [markdown]
# Grafik, kitabın kullandığı `ceterisParibus` paketinin görünümüyle çizilmiştir: gri CP eğrisi, siyah nokta (gözlemin kendisi),
# test verisi aralığında 101 noktalı ızgara.
#
# > *Şekil (kitap, `fig-ceteris-penguins`): Tek bir penguen ve gaga derinliği için ceteris paribus grafiği. Çizgi, gaga derinliği değiştirildiğinde belirli bir penguen için tahmin edilen değeri gösterir. Penguenin gerçek gaga derinliği bir noktayla işaretlenmiştir.*

# %%
fig, ax = plt.subplots(figsize=(8.5, 6))
cp_ciz(ax, p_disi, penguen, "bill_depth_mm", sayisal_izgara(p_test["bill_depth_mm"], 101), renk=GRI_CIZGI, nokta_renk="black")
ax.set(xlabel="Gaga derinliği (mm)", ylabel="P(dişi)", ylim=(-0.03, 1.03), yticks=[0, 0.25, 0.5, 0.75, 1])
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
drwhy(ax)
drwhy_baslik(ax, "bill_depth_mm")
plt.show()

# %% [markdown]
# Bu Python modelinde gaga derinliği 20,6 mm'den yaklaşık 20 mm'ye doğru azalırken P(dişi) önce biraz düşüyor.
# Daha sığ gaga değerlerine gidildiğinde ise genel olarak belirgin biçimde yükseliyor. Bu yön Türkçe çevirideki
# P(dişi) eğrisiyle uyumlu. P(erkek), P(dişi) olasılığının tümleyenidir.

# %% [markdown]
# ### 2.2 Kitaptaki uyarı: "İkili bağımlılıkların ötesine bak"
#
# En hafif Gentoo pengueninin gaga derinliğini 3,3 mm artırıyoruz. Nokta (gaga derinliği, kütle) ikilisi için gerçekçidir,
# (gaga derinliği, tür) ikilisi için de gerçekçidir; ama üçü birlikte düşünüldüğünde gerçekçi değildir.
#
# > *Şekil (kitap, `fig-pairwise-not-enough`): Gaga derinliği ve vücut kütlesinin saçılım grafiği. Bir penguenin gaga derinliğini manipüle etmek gerçekçi olmayan bir veri noktası üretebilir.*

# %%
gentoo = peng[peng["species"] == "Gentoo"].nsmallest(1, "body_mass_g").iloc[0]
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.annotate("", xy=(gentoo["bill_depth_mm"] + 3.3, gentoo["body_mass_g"]), xytext=(gentoo["bill_depth_mm"], gentoo["body_mass_g"]),
            arrowprops=dict(arrowstyle="->", lw=2, color="black", mutation_scale=28, shrinkA=0, shrinkB=0))
ax.scatter(gentoo["bill_depth_mm"], gentoo["body_mass_g"], s=150, color="red", zorder=2)    # önce kırmızı halka
for tur, r in zip(["Adelie", "Chinstrap", "Gentoo"], VIRIDIS_3):                           # sonra bütün noktalar üstte
    d = peng[peng["species"] == tur]
    ax.scatter(d["bill_depth_mm"], d["body_mass_g"], s=16, color=r, label=tur, zorder=3)
ax.set(xlabel="Gaga derinliği (mm)", ylabel="Vücut kütlesi (g)")
ax.legend(title="species", loc="center left", bbox_to_anchor=(1.02, 0.5), title_fontsize=13, fontsize=11.5, markerscale=1.3)
plt.show()

# %% [markdown]
# ### 2.3 Bisiklet: tek bir gün için tüm özniteliklerin what-if analizi (SVM)
#
# R'deki `e1071::svm` girdileri ve hedefi varsayılan olarak ölçeklendirir; aynısını `StandardScaler` +
# `TransformedTargetRegressor` ile yapıyorum. Kitabın açıkladığı 3 Ocak 2011 gününü eğitim dışında tutuyorum.
# Kalan günleri Python'da sabit tohumla ayırıyorum.

# %%
bike = veri_oku("bike.csv")
BIKE_OZ = ["season", "holiday", "workday", "weather", "temp", "hum", "windspeed", "cnt_2d_bfr"]
BIKE_KAT = ["season", "holiday", "workday", "weather"]
BIKE_SAY = ["temp", "hum", "windspeed", "cnt_2d_bfr"]
KITAP_GUNU = "2011-01-03"
bike = bike[["dteday"] + BIKE_OZ + ["cnt"]].dropna().set_index("dteday")
kitap_gunu = bike.loc[[KITAP_GUNU]]
b_egitim, b_test = train_test_split(
    bike.drop(index=KITAP_GUNU), train_size=len(bike) * 2 // 3, random_state=42
)
b_test = pd.concat([kitap_gunu, b_test])
assert KITAP_GUNU not in b_egitim.index
on_isleme = lambda: ColumnTransformer([("kat", OneHotEncoder(handle_unknown="ignore"), BIKE_KAT),
                                      ("say", StandardScaler(), BIKE_SAY)])
bike_modeller = {
    "Doğrusal model": Pipeline([("on", on_isleme()), ("m", LinearRegression())]),
    "Karar ağacı": Pipeline([("on", on_isleme()), ("m", DecisionTreeRegressor(max_depth=4, min_samples_leaf=7, random_state=42))]),
    "Rastgele orman": Pipeline([("on", on_isleme()), ("m", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1))]),
    "SVM": TransformedTargetRegressor(Pipeline([("on", on_isleme()), ("m", SVR(C=1.0))]), transformer=StandardScaler()),
}
for ad, m in bike_modeller.items():
    m.fit(b_egitim[BIKE_OZ], b_egitim["cnt"])
    print(f"{ad:15s} test R² = {r2_score(b_test['cnt'], m.predict(b_test[BIKE_OZ])):.3f}")

gun = kitap_gunu[BIKE_OZ].reset_index(drop=True)
svm = bike_modeller["SVM"].predict
print(f"\nİncelenen gün: {KITAP_GUNU}, {gun['season'][0]}, hava {gun['weather'][0]}, "
      f"sıcaklık {gun['temp'][0]:.1f} °C, SVM tahmini {svm(gun)[0]:.0f}, "
      f"gerçek kiralama sayısı {kitap_gunu['cnt'].iloc[0]}")

# %% [markdown]
# Grafik DALEX'in `predict_profile` çıktısının görünümüyle çizilmiştir: değişkenler alfabetik sırada, kategorik çubuklar gözlemin
# tahmininden (kesikli çizgi) başlar, gözlemin gerçek değeri mor noktayla işaretlidir.
#
# > *Şekil (kitap, `fig-whatif-bike`): Tekil özniteliklerin değiştirilmesinin tahmin edilen bisiklet kiralama sayısını nasıl değiştirdiği.*

# %%
def dalex_duzen(alt_sekil, model_adi, sol=0.12, ara=0.35):
    """DALEX predict_profile düzeni: büyük başlık + alt başlık sol üstte, paneller altında."""
    alt_sekil.subplots_adjust(left=sol, right=0.97, top=0.83, bottom=0.09, wspace=ara, hspace=0.32)
    alt_sekil.text(0.02, 0.985, "Ceteris Paribus profile", ha="left", va="top", color=MOR, fontsize=18)
    alt_sekil.text(0.02, 0.925, f"created for the {model_adi} model", ha="left", va="top", color=MOR, fontsize=12.5)


fig = plt.figure(figsize=(14, 8.5))
alt = fig.subfigures(1, 2)
ax_k = alt[0].subplots(2, 2, sharex=True)
for ax, oz in zip(ax_k.ravel(), sorted(BIKE_KAT)):
    cp_cubuk(ax, svm, gun, oz, bike[oz].unique())
alt[0].supxlabel("prediction", color=MOR, fontsize=13)
dalex_duzen(alt[0], "svm", sol=0.13, ara=0.45)
ax_s = alt[1].subplots(2, 2, sharey=True)
for ax, oz in zip(ax_s.ravel(), sorted(BIKE_SAY)):
    cp_ciz(ax, svm, gun, oz, sayisal_izgara(b_test[oz], 101))
    drwhy(ax)
    drwhy_baslik(ax, oz)
alt[1].supylabel("prediction", color=MOR, fontsize=13)
dalex_duzen(alt[1], "svm", sol=0.12, ara=0.12)
plt.show()

# %% [markdown]
# ### 2.4 Sparklines: minimal sürüm (kitaptaki ipucu kutusu)
#
# Kitaptaki gibi ggplot2 `theme_void`: eksen ve ızgara yok, ortak y ekseni.

# %%
fig, eksenler = plt.subplots(1, 4, figsize=(14, 1.9), sharey=True)
for ax, oz in zip(eksenler, sorted(BIKE_SAY)):
    iz = sayisal_izgara(b_test[oz], 101)
    ax.plot(iz, cp_profili(svm, gun, oz, iz), color="black", lw=1.1)
    ax.scatter(gun[oz], svm(gun), color="black", s=22, zorder=5)
    ax.set_title(oz.replace("_", "  "), fontsize=12)
    ax.axis("off")
plt.show()

# %% [markdown]
# ### 2.5 Aynı gün, dört farklı model: sıcaklık CP'si
#
# Model adları ve renk/çizgi tipleri kitaptaki gibidir (R'deki etiketler: `lm`, `randomForest`, `svm`, `tree`).
#
# > *Şekil (kitap, `fig-cp-bike-models`): Bisiklet tahmin görevi için farklı modellere ait ceteris paribus eğrileri: doğrusal model, rastgele orman, SVM ve karar ağacı.*

# %%
R_ETIKET = {"lm": "Doğrusal model", "randomForest": "Rastgele orman", "svm": "SVM", "tree": "Karar ağacı"}
fig, ax = plt.subplots(figsize=(8.5, 6.2))
iz = sayisal_izgara(b_test["temp"], 101)
for etiket, (renk_m, cizgi) in GG_MODEL.items():
    ax.plot(iz, cp_profili(bike_modeller[R_ETIKET[etiket]].predict, gun, "temp", iz), color=renk_m, ls=cizgi, lw=2.2, label=etiket)
ax.set(xlabel="Sıcaklık (°C)", ylabel="Tahmin edilen bisiklet kiralama sayısı")
drwhy(ax)
drwhy_baslik(ax, "temp")
cizgiler, etiketler = ax.get_legend_handles_labels()
baslik_tutamac = plt.Line2D([], [], color="none")                 # ggplot2 gibi: "Model" başlığı efsanenin solunda
ax.legend([baslik_tutamac] + cizgiler, ["Model"] + etiketler, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.08),
          handlelength=3, fontsize=11.5)
plt.show()

# %% [markdown]
# ## 3. Asıl uygulama: IBM Telco Customer Churn
#
# **Veri:** IBM'in örnek veri seti; bir telekom şirketinin 7.043 müşterisi, 19 öznitelik (sözleşme tipi, abonelik süresi,
# aylık ve toplam ücret, internet ve ek hizmetler, ödeme yöntemi, demografi) ve hedef: müşteri son ay içinde ayrıldı mı?
#
# CP bu veri için doğal bir araçtır, çünkü bir müşteri temsilcisi tam olarak şunu sorar: *"Bu müşteri iki yıllık sözleşmeye
# geçse ya da faturası 20 $ düşse ayrılma riski ne olur?"*
#
# **Bu veri neden zorlu?**
#
# | Sorun | CP açısından anlamı |
# |---|---|
# | `TotalCharges ≈ tenure × MonthlyCharges` (yaklaşık ilişki) | `tenure` değişirken `TotalCharges` sabit tutulduğunda eğitim verisinden uzak senaryolar oluşabilir |
# | `InternetService = "No"` ise 6 ek hizmet sütunu zorunlu olarak `"No internet service"` olur | Kategorik CP, verinin hiçbir zaman içermediği kombinasyonlar üretir |
# | Dengesiz sınıflar (%26,5 churn) | Olasılık ölçeğinde küçük görünen değişimler göreli olarak büyüktür |
# | `TotalCharges`'ta 11 boş değer (yeni müşteriler, `tenure = 0`) | Temizleme kararı gerekir (0 olarak dolduruldu) |
# | Sözleşme, ödeme yöntemi, internet tipi birbiriyle güçlü ilişkili | Tek bir kategoriyi değiştirmek de nadir müşteri profilleri yaratır |

# %%
tel = veri_oku("telco_churn.csv").set_index("customerID")
tel["TotalCharges"] = pd.to_numeric(tel["TotalCharges"], errors="coerce").fillna(0.0)
tel["Churn"] = (tel["Churn"] == "Yes").astype(int)
T_SAY = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
T_KAT = [c for c in tel.columns if c not in T_SAY + ["Churn"]]
T_OZ = T_SAY + T_KAT
HIZMETLER = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

print(tel.shape, f"churn oranı = {tel['Churn'].mean():.1%}")
oran = tel["TotalCharges"] / (tel["tenure"] * tel["MonthlyCharges"]).replace(0, np.nan)
print(f"corr(tenure, TotalCharges) = {tel['tenure'].corr(tel['TotalCharges']):.2f};  "
      f"TotalCharges / (tenure × MonthlyCharges) oranının %5–%95 aralığı: {oran.quantile(0.05):.2f} – {oran.quantile(0.95):.2f}")

# %%
MUSTERI_ID = "5949-XIKAE"                       # açıklanacak demo müşterisi (Bölüm 3.1), sabit kayıt
t_egitim, t_test = train_test_split(tel.drop(index=MUSTERI_ID), train_size=2 / 3, random_state=42,
                                   stratify=tel.drop(index=MUSTERI_ID)["Churn"])
t_test = pd.concat([tel.loc[[MUSTERI_ID]], t_test])  # demo müşterisi her zaman test kümesinde, eğitimde asla
assert MUSTERI_ID not in t_egitim.index
kat_on = lambda: ColumnTransformer([("kat", OneHotEncoder(handle_unknown="ignore"), T_KAT),
                                   ("say", StandardScaler(), T_SAY)], sparse_threshold=0)
churn_modeller = {
    "Gradient boosting": Pipeline([("on", kat_on()), ("m", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))]),
    "Rastgele orman": Pipeline([("on", kat_on()), ("m", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1))]),
    "Lojistik regresyon": Pipeline([("on", kat_on()), ("m", LogisticRegression(max_iter=2000))]),
}
for ad, m in churn_modeller.items():
    m.fit(t_egitim[T_OZ], t_egitim["Churn"])
    print(f"{ad:20s} test ROC-AUC = {roc_auc_score(t_test['Churn'], m.predict_proba(t_test[T_OZ])[:, 1]):.3f}")

IZ_TENURE = np.arange(0, 73)                   # tenure tam sayı aydır: bütün tenure grafiklerinde 0–72 ay
gb = churn_modeller["Gradient boosting"]
p_churn = lambda D: gb.predict_proba(D[T_OZ])[:, 1]

# %% [markdown]
# ### 3.1 Açıklanacak müşteriyi seçmek
#
# Gerçekçi bir senaryo: müşteri hizmetleri ekibi **yüksek riskli, yeni** bir müşteriyi ele alıyor. Müşteri `5949-XIKAE`:
# aylık sözleşmeli, fiber internetli, 8 aydır abone, elektronik çekle ödüyor. Bu müşteriyi **sabit kimliğiyle** seçiyorum
# (başlangıçta "modelin riskini yaklaşık 0,70 tahmin ettiği yeni bir fiber müşterisi" ölçütüyle bulunmuştu). Böylece kütüphane
# sürümü ya da model değişse bile açıklanan kişi değişmez; değişebilecek olan yalnızca modelin bu kişi için verdiği sayılardır.

# %%
musteri = t_test.loc[[MUSTERI_ID], T_OZ].reset_index(drop=True)
p0 = p_churn(musteri)[0]
print(f"Müşteri {MUSTERI_ID}: tenure = {musteri['tenure'][0]} ay, aylık {musteri['MonthlyCharges'][0]} $, "
      f"gerçek durum: {'ayrıldı' if t_test.loc[MUSTERI_ID, 'Churn'] else 'kaldı'}; modelin tahmini P(ayrılma) = {p0:.2f}")

# Kategorik öznitelik için de doğrulama: CP kodu ile bağımsız referans döngü aynı sonucu vermeli
sozlesmeler = ["Month-to-month", "One year", "Two year"]
kat_hizli = cp_profili(p_churn, musteri, "Contract", sozlesmeler)
kat_ref = cp_referans(p_churn, musteri, "Contract", sozlesmeler)
print(pd.DataFrame({"Contract": sozlesmeler, "cp_profili": kat_hizli, "referans döngü": kat_ref}).round(4).to_string(index=False))
assert np.allclose(kat_hizli, kat_ref)
print("✔ Kategorik öznitelik: 3 sözleşme tipinde cp_profili = bağımsız referans")
print(musteri.T.rename(columns={0: "değer"}).to_string())

# %% [markdown]
# ### 3.2 Müşteri için what-if profili (kitaptaki bisiklet şeklinin karşılığı)

# %%
KAT_GOSTER = ["Contract", "InternetService", "OnlineSecurity", "PaperlessBilling", "PaymentMethod", "TechSupport"]
fig = plt.figure(figsize=(15, 10))
alt = fig.subfigures(1, 2, width_ratios=[1.25, 1])
ax_k = alt[0].subplots(3, 2, sharex=True)
for ax, oz in zip(ax_k.ravel(), KAT_GOSTER):
    cp_cubuk(ax, p_churn, musteri, oz, tel[oz].unique())
    ax.set_xlim(0, 1)
alt[0].supxlabel("P(ayrılma)", color=MOR, fontsize=13)
dalex_duzen(alt[0], "gradient boosting", sol=0.2, ara=0.85)
ax_s = alt[1].subplots(3, 1, sharey=True)
for ax, oz in zip(ax_s, ["MonthlyCharges", "tenure", "TotalCharges"]):
    cp_ciz(ax, p_churn, musteri, oz, IZ_TENURE if oz == "tenure" else sayisal_izgara(tel[oz], 101))
    ax.set_ylim(0, 1)
    drwhy(ax)
    drwhy_baslik(ax, oz)
alt[1].supylabel("P(ayrılma)", color=MOR, fontsize=13)
dalex_duzen(alt[1], "gradient boosting", sol=0.12)
plt.show()

senaryo = {}
for oz, deger in [("Contract", "One year"), ("Contract", "Two year"), ("PaymentMethod", "Credit card (automatic)"),
                  ("TechSupport", "Yes"), ("OnlineSecurity", "Yes")]:
    senaryo[f"{oz} = {deger}"] = cp_profili(p_churn, musteri, oz, [deger])[0]
print(f"Mevcut P(ayrılma) = {p0:.2f}")
print(pd.Series(senaryo, name="P(ayrılma)").round(2))

# %% [markdown]
# *(Bu bölümdeki ve aşağıdaki sayılar `requirements.txt` dosyasındaki paket sürümleriyle yapılan çalıştırmaya aittir.)*
#
# Model girdisindeki sözleşme türü aylıktan iki yıllığa alındığında tahmin 0,62'den 0,07'ye iniyor.
# Otomatik kredi kartı ödemesi ve çevrimiçi güvenlik senaryolarında model sırasıyla 0,52 ve 0,49 veriyor.
# Bunlar sabit modeldeki girdi değişimlerine verilen tepkilerdir. Nedensel etkiyi ölçmek ayrı bir analiz gerektirir.
#
# Klasik CP'de `tenure` arttıkça ayrılma tahmini genel olarak düşüyor. `TotalCharges` eğrisinde belirgin bir yön yok.
# Bu profilleri, iki değişkenin veride birlikte değiştiğini dikkate alarak okumak gerekir.

# %% [markdown]
# ### 3.3 Kitabın ötesi: aynı müşteri için farklı sözleşmeler ve modeller
#
# `tenure` CP eğrisini iki biçimde karşılaştırıyoruz. Önce aynı müşterinin üç sözleşme senaryosunu tek modelde,
# ardından müşterinin mevcut sözleşmesini üç farklı modelde inceliyoruz.

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
iz_t = IZ_TENURE
for sozlesme, r in zip(["Month-to-month", "One year", "Two year"], ["#d62728", "#ff7f0e", "#2ca02c"]):
    biri = musteri.copy()
    biri["Contract"] = sozlesme
    cp_ciz(a1, p_churn, biri, "tenure", iz_t, renk=r, etiket=sozlesme, nokta=(sozlesme == "Month-to-month"))
a1.set(xlabel="tenure (ay)", ylabel="P(ayrılma)", title="Aynı müşteri, üç sözleşme tipi", ylim=(0, 1)); a1.legend()
for (ad, m), stil in zip(churn_modeller.items(), ["-", "--", ":"]):
    a2.plot(iz_t, cp_profili(lambda D: m.predict_proba(D[T_OZ])[:, 1], musteri, "tenure", iz_t), lw=2.2, ls=stil, label=ad)
a2.set(xlabel="tenure (ay)", title="Aynı müşteri, üç model"); a2.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4 Sınırlılık: sayısal CP eğitim verisinden uzak senaryolar üretebilir
#
# Klasik CP'de müşterinin `tenure` değeri 0–72 ay arasında değiştirilirken `TotalCharges` sabit kalır.
# Bu senaryoların eğitim verisindeki ödeme örüntüsünden ne kadar uzaklaştığını iki şekilde inceliyorum.
#
# 1. **Haritada:** CP'nin izlediği yol `tenure`–`TotalCharges` saçılımında verinin tamamen boş olduğu bölgeden geçer.
# 2. **Gerçekçilik testi:** 3 sayısal öznitelik standartlaştırılır; eğitim verisinde en yakın komşu uzaklıklarının %99 kantili
#    $\tau$ eşiğidir. CP noktası en yakın gerçek müşteriye $\tau$'dan uzaksa **veri dışı (OOD)** sayılır.

# %%
GERCEK_OZ = ["tenure", "MonthlyCharges", "TotalCharges"]
olcek = StandardScaler().fit(t_egitim[GERCEK_OZ])
ref = olcek.transform(t_egitim[GERCEK_OZ])
nn = NearestNeighbors(n_neighbors=2).fit(ref)
tau = np.quantile(nn.kneighbors(ref)[0][:, 1], 0.99)


def ood(D):
    return nn.kneighbors(olcek.transform(D[GERCEK_OZ]), n_neighbors=1)[0][:, 0] > tau


def yol(x, ozellik, izgara, tutarli=False):
    """CP'nin oluşturduğu yapay müşteriler. tutarli=True ise TotalCharges = tenure × MonthlyCharges ile güncellenir."""
    D = pd.concat([x] * len(izgara), ignore_index=True)
    D[ozellik] = izgara
    if tutarli:
        D["TotalCharges"] = D["tenure"] * D["MonthlyCharges"] * (x["TotalCharges"].iloc[0] / max(x["tenure"].iloc[0] * x["MonthlyCharges"].iloc[0], 1e-9))
    return D


naif, tutarli = yol(musteri, "tenure", iz_t), yol(musteri, "tenure", iz_t, tutarli=True)
print(f"τ = {tau:.3f}; naif tenure CP noktalarının %{100 * ood(naif).mean():.0f}'i veri dışı, "
      f"tutarlı CP noktalarının %{100 * ood(tutarli).mean():.0f}'i veri dışı")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(t_egitim["tenure"], t_egitim["TotalCharges"], s=3, color="0.8", label="gerçek müşteriler")
o = ood(naif)
ax.scatter(naif["tenure"][~o], naif["TotalCharges"][~o], s=22, color="#1f77b4", label="naif CP: gerçekçi")
ax.scatter(naif["tenure"][o], naif["TotalCharges"][o], s=22, color="#d62728", label="naif CP: uzaklık eşiğinin üzerinde")
ax.plot(tutarli["tenure"], tutarli["TotalCharges"], color="#2ca02c", lw=2.5, label="tutarlı CP yolu")
ax.scatter(musteri["tenure"], musteri["TotalCharges"], s=200, marker="*", color="gold", edgecolor="k", zorder=5, label="müşteri")
ax.set(xlabel="tenure (ay)", ylabel="TotalCharges ($)", title="Naif CP, gerçek müşterilerin olmadığı bölgeden geçiyor")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ### 3.5 Bu sadece estetik bir sorun mu? Naif ve tutarlı CP karşılaştırması
#
# Bir **düzeltme** öneriyorum: `tenure` (ya da `MonthlyCharges`) değiştiğinde `TotalCharges`'ı da müşterinin kendi
# ödeme oranını koruyarak $\text{TotalCharges} = \text{tenure} \times \text{MonthlyCharges} \times r_i$ ile güncellemek.
# Bu, değişkenler arasındaki **bilinen yapısal ilişkiyi** CP'ye dahil eder ve klasik *ceteris paribus* varsayımının yerine
# "*diğer her şey, mantıksal olarak bağlı olanlar hariç, eşitken*" varsayımını koyar.

# %%
fig, eksenler = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
fark = {}
for ax, oz in zip(eksenler, ["tenure", "MonthlyCharges"]):
    iz = IZ_TENURE if oz == "tenure" else sayisal_izgara(tel[oz], 60)
    y_naif = p_churn(yol(musteri, oz, iz))
    y_tut = p_churn(yol(musteri, oz, iz, tutarli=True))
    o_naif = ood(yol(musteri, oz, iz))
    ax.plot(iz, y_naif, color="#d62728", lw=2, label="naif CP (TotalCharges sabit)")
    ax.scatter(iz[o_naif], y_naif[o_naif], color="#d62728", s=10)
    ax.plot(iz, y_tut, color="#2ca02c", lw=2.5, label="tutarlı CP (TotalCharges güncellenir)")
    ax.scatter(musteri[oz], p0, s=80, color="gold", edgecolor="k", zorder=5)
    ax.set(xlabel=oz, title=f"{oz}: naif ve tutarlı CP", ylim=(0, 1))
    fark[oz] = {"en büyük fark": np.abs(y_naif - y_tut).max(),
                f"{oz} max'ta naif P": y_naif[-1], f"{oz} max'ta tutarlı P": y_tut[-1],
                "naif OOD oranı": o_naif.mean()}
eksenler[0].set_ylabel("P(ayrılma)"); eksenler[0].legend(fontsize=8)
fig.tight_layout()
plt.show()
for oz, d in fark.items():
    print(oz, {k: round(float(v), 2) for k, v in d.items()})

# Örnek aylardaki model tahminlerini ve uzaklık kontrolünü yan yana göster.
y_naif_t = p_churn(yol(musteri, "tenure", IZ_TENURE))
y_tut_t = p_churn(yol(musteri, "tenure", IZ_TENURE, tutarli=True))
ood_naif_t = ood(yol(musteri, "tenure", IZ_TENURE))
ood_tut_t = ood(yol(musteri, "tenure", IZ_TENURE, tutarli=True))
print(pd.DataFrame({"tenure": IZ_TENURE, "naif P": y_naif_t, "naif OOD": ood_naif_t,
                    "tutarlı P": y_tut_t, "tutarlı OOD": ood_tut_t}).set_index("tenure").loc[[38, 65, 66, 72]].round(2))

# %% [markdown]
# **Yorum:**
#
# * İki eğri müşterinin gerçek değeri olan 8 ayda aynı girdiyi kullanır.
# * Senaryolar ilerledikçe, tahminler farklılaşır. 38. ayda klasik CP 0,41, toplam ödemenin güncellendiği senaryo 0,63 verir.
# * Uzaklık kontrolü eğitim verisine yakınlığı ölçer. İlişkili öznitelikleri birlikte güncelleyen eğri,
#   klasik CP'den farklı bir senaryoyu temsil eder ve tahminler bu çerçevede yorumlanır.
#
# %% [markdown]
# ### 3.6 Zafiyet 2: kategorik CP ve mantıksal kısıtlar
#
# Naif kategorik CP `InternetService`'i `"No"` yaptığında müşterinin `OnlineSecurity`, `StreamingTV` vb. sütunları hâlâ
# `"Yes"`/`"No"` kalır. Oysa veride internet hizmeti olmayan **her** müşteride bu sütunlar `"No internet service"`'tir.
# Model bu kombinasyonu **hiç görmemiştir**; yine de bir olasılık üretir.

# %%
def tutarli_internet(x, deger):
    y = x.copy()
    y["InternetService"] = deger
    if deger == "No":
        y[HIZMETLER] = "No internet service"
    elif (x[HIZMETLER] == "No internet service").any(axis=None):
        y[HIZMETLER] = "No"
    return y


satirlar = []
for deger in ["DSL", "Fiber optic", "No"]:
    naif_m = musteri.copy(); naif_m["InternetService"] = deger
    tut_m = tutarli_internet(musteri, deger)
    egitimde = (t_egitim[["InternetService"] + HIZMETLER] == naif_m[["InternetService"] + HIZMETLER].iloc[0]).all(axis=1).sum()
    satirlar.append({"InternetService": deger, "naif P(ayrılma)": p_churn(naif_m)[0], "tutarlı P(ayrılma)": p_churn(tut_m)[0],
                     "naif kombinasyonun eğitim verisindeki sayısı": egitimde})
kategorik_tablo = pd.DataFrame(satirlar).set_index("InternetService")
print(kategorik_tablo.round(3))

fig, ax = plt.subplots(figsize=(7, 3))
kategorik_tablo[["naif P(ayrılma)", "tutarlı P(ayrılma)"]].plot.barh(ax=ax, color=["#d62728", "#2ca02c"])
ax.set(xlabel="P(ayrılma)", xlim=(0, 1), title="InternetService için kategorik CP: naif ve tutarlı")
plt.show()

# %% [markdown]
# ### 3.7 Kararlılık: bootstrap örneklemleriyle yeniden eğitim
#
# Eğitim kümesinden yerine koyarak 10 bootstrap örneklemi çekiyoruz. Her örneklemde modeli yeniden eğitip
# aynı müşteri için CP eğrisini hesaplıyoruz.

# %%
fig, ax = plt.subplots(figsize=(8, 4.3))
egriler = []
for tohum in range(10):
    orneklem = t_egitim.sample(frac=1, replace=True, random_state=tohum)
    m = Pipeline([("on", kat_on()), ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                          l2_regularization=1.0, random_state=tohum))])
    m.fit(orneklem[T_OZ], orneklem["Churn"])
    egriler.append(cp_profili(lambda D: m.predict_proba(D[T_OZ])[:, 1], musteri, "tenure", iz_t))
egriler = np.array(egriler)
ax.plot(iz_t, egriler.T, color="#1f77b4", alpha=0.4)
ax.fill_between(iz_t, egriler.min(0), egriler.max(0), color="#1f77b4", alpha=0.12, label="10 bootstrap modelinin aralığı")
ax.axvline(musteri["tenure"][0], color="gold", lw=2, ls="--", label="müşterinin gerçek tenure değeri")
ax.set(xlabel="tenure (ay)", ylabel="P(ayrılma)", ylim=(0, 1), title="CP'nin model yeniden eğitimine duyarlılığı")
ax.legend(fontsize=8)
plt.show()
genislik = egriler.max(0) - egriler.min(0)
yakin = np.abs(iz_t - musteri["tenure"][0]) <= 3
print(f"Bant genişliği: gerçek değerin ±3 ay yakınında ortalama {genislik[yakin].mean():.2f}, "
      f"uzak bölgelerde ortalama {genislik[~yakin].mean():.2f}")

# %% [markdown]
# **Yorum.** Bant, 10 yeniden eğitimde her ızgara noktası için gözlenen en düşük ve en yüksek tahmini gösterir.
# Bu çalıştırmada eğriler, müşterinin gözlenen abonelik süresine yakın değerlerde de farklılaşıyor.

# %% [markdown]
# ## 4. Özet
#
# | Bulgu | Kanıt |
# |---|---|
# | Sıfırdan yazılan CP doğru | 6 gaga derinliği ve 3 sözleşme tipinde bağımsız referans döngüyle birebir aynı (Bölüm 2.1, 3.1) |
# | Kitaptaki şekiller Python'da yeniden üretildi | Kitaptaki aynı Adelie pengueni (20,6 mm) dahil, Bölüm 2.1 – 2.5 |
# | CP, modelin tek bir müşteriye ilişkin senaryo tahminlerini gösteriyor | İki yıllık sözleşme girdisinde model tahmini 0,62’den 0,07’ye iniyor (Bölüm 3.2) |
# | Modeller aynı müşteri için farklı CP şekilleri öğreniyor | Bölüm 3.3 |
# | Klasik sayısal CP eğitim verisinden uzak noktalar üretiyor | Uzaklık eşiğini klasik CP’de %90, toplam ödeme güncellendiğinde %0 aşıyor (Bölüm 3.4) |
# | İki senaryo farklı model tahminleri üretiyor | Genel eğilim aynı kalsa da naif ve tutarlı CP 0,22'ye kadar ayrışıyor (ör. 38. ayda 0,41 ve 0,63) (Bölüm 3.5) |
# | Kategorik CP mantıksal kısıtları çiğniyor | Naif `InternetService = "No"` kombinasyonu eğitim verisinde 0 kez var; naif P = 0,32, tutarlı P = 0,22 (Bölüm 3.6) |
# | CP model yeniden eğitimine duyarlı | 10 bootstrap modelinin bandı müşterinin kendi değerinin ±3 ay yakınında bile ortalama 0,26 genişliğinde (Bölüm 3.7) |
#
# Telco örneğinde CP eğrisini yorumlarken hangi girdilerin birlikte değiştiğine ve senaryoların eğitim verisindeki
# desteğine bakıyoruz. Bootstrap örneklemleri de aynı müşteri için eğrinin yeniden eğitime duyarlılığını gösteriyor.
