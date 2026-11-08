# %% [markdown]
# # Bölüm 13 – Bireysel Koşullu Beklenti (ICE): Uygulama
#
# **Öğrenci:** Serap Şahin (20269348009)
# **Kaynak bölüm:** Molnar, *Interpretable Machine Learning*, Bölüm 13 – Individual Conditional Expectation (ICE)
#
# Bu uygulama üç kısımdan oluşur:
#
# 1. **ICE'yi sıfırdan yazmak.** ICE, c-ICE ve d-ICE hesapları `numpy` ve `pandas` ile yapılır.
#    Temel ICE matrisi `scikit-learn` ile aynı ızgarada karşılaştırılır.
# 2. **Kitabın örneklerini Python'a uyarlamak.** Bisiklet ve penguen grafikleri yeniden çizilir.
# 3. **Asıl uygulama: California Housing.** PDP'nin ortalamasında gizlenen model davranışı ve ICE ızgarasının
#    eğitim verisinden uzak kombinasyonlar üretmesi incelenir.
#
# Tanım (kitaptaki gösterimle): $i$. örnek ve $j$. öznitelik için ICE eğrisi
#
# $$\hat{f}^{(i)}_j(z) = \hat{f}\big(z, \mathbf{x}^{(i)}_{-j}\big), \qquad z \in \{z_1,\dots,z_K\}$$
#
# PDP ise bu eğrilerin ortalamasıdır: $\hat{f}_{j}^{PDP}(z) = \frac{1}{n}\sum_{i=1}^{n}\hat{f}^{(i)}_j(z)$.

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# ---- Kitaptaki grafiklerin görünümü: ggplot2 theme_bw (kitap R'de iml + ggplot2 kullanır) ----
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
IML_IZGARA = 20                                   # iml paketinin varsayılan ızgara büyüklüğü
VIRIDIS_4 = ["#440154", "#31688E", "#35B779", "#FDE725"]   # ggplot2 scale_color_viridis(discrete = TRUE), 4 sınıf


def rug(ax, x, renk="black"):
    """ggplot2 geom_rug: eksenin altında veri noktalarının dağılımı."""
    ax.vlines(np.asarray(x), 0, 0.03, transform=ax.get_xaxis_transform(), color=renk, alpha=0.25, lw=0.7)


def efsane_sag(ax, baslik):
    """ggplot2 tarzı, grafiğin sağında çerçevesiz efsane."""
    ax.legend(title=baslik, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title_fontsize=13, fontsize=11.5)

URL = {
    "bike.csv": "https://raw.githubusercontent.com/christophM/interpretable-ml-book/8f470c85e5ed4ebd69de74425cb9832f1bdc25eb/data/bike.csv",
    "penguins.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/71e2436a092d714350de0fc409ca8a8714e7e78f/penguins.csv",
    "housing.csv": "https://raw.githubusercontent.com/ageron/handson-ml2/7b7e23e7267356f8355580877eff98c43cda1bd0/datasets/housing/housing.csv",
}


def veri_oku(ad):
    """Veri setini GitHub'dan indirir (internet bağlantısı gerekir)."""
    return pd.read_csv(URL[ad])



# %% [markdown]
# ## 1. ICE'yi sıfırdan yazmak
#
# Algoritma çok basittir: $n$ örneği $K$ kez kopyala, her kopyada $j$. özniteliği ızgaranın bir değeriyle değiştir,
# modelden tahmin al ve sonucu $n \times K$ boyutlu bir matrise yerleştir. Satırlar ICE eğrileridir,
# sütun ortalaması PDP'dir.
#
# * **c-ICE (merkezlenmiş):** $\hat{f}^{(i)}_j(z) - \hat{f}^{(i)}_j(a)$; $a$ referans (anchor) noktası, genellikle ızgaranın alt ucu.
# * **d-ICE (türev):** $\partial \hat{f}^{(i)}_j(z) / \partial z$; sayısal olarak sonlu farklarla hesaplanır.
#   Etkileşim yoksa $\hat f(\mathbf{x}) = g(x_j) + h(\mathbf{x}_{-j})$ olur ve tüm türev eğrileri $g'(x_j)$ ile çakışır.
#   Türevlerin standart sapması bu yüzden **etkileşimin yerel bir ölçüsüdür**.

# %%
def ice_hesapla(tahmin_fn, X, ozellik, izgara):
    """n x K boyutlu ICE matrisini döndürür (satır = örnek, sütun = ızgara noktası)."""
    n, K = len(X), len(izgara)
    X_cogul = pd.concat([X] * K, ignore_index=True)        # ızgara-blok sırası: önce z_1 için n satır, sonra z_2 ...
    X_cogul[ozellik] = np.repeat(np.asarray(izgara), n)
    return np.asarray(tahmin_fn(X_cogul)).reshape(K, n).T


def merkezle(ice, referans_sutunu=0):
    return ice - ice[:, [referans_sutunu]]


def turev(ice, izgara):
    return np.gradient(ice, np.asarray(izgara, dtype=float), axis=1)


def izgara_olustur(x, K=50, alt=0.0, ust=1.0):
    """Kitaptaki gibi eşit aralıklı ızgara; alt/ust kantil verilirse aralık kırpılır."""
    return np.linspace(np.quantile(x, alt), np.quantile(x, ust), K)


def ice_ciz(ax, izgara, ice, renk="black", alfa=0.2, pdp=True, etiketler=None, renkler=None):
    if renkler is None:
        ax.plot(izgara, ice.T, color=renk, alpha=alfa, lw=0.6)
    else:
        for kategori, r in renkler.items():
            m = etiketler == kategori
            ax.plot(izgara, ice[m].T, color=r, alpha=alfa, lw=0.7)
            ax.plot([], [], color=r, lw=1.2, alpha=0.6, label=kategori)
    if pdp:
        ax.plot(izgara, ice.mean(axis=0), color="#d62728", lw=3, label="PDP (ortalama)")


# %% [markdown]
# ## 2. Kitabın örneklerini yeniden üretmek
#
# Kitaptaki öznitelikler, eğitim/test oranı ve rastgele orman yaklaşımı korunur.
# Python'daki veri ayrımı ve model uygulaması eğrilerin ayrıntılarını değiştirebilir.
# (R'deki `randomForest` yerine `sklearn` kullanıldığı için sayılar birebir aynı olmaz, şekiller aynı davranışı gösterir.)

# %%
bike = veri_oku("bike.csv")
BIKE_OZ = ["season", "holiday", "workday", "weather", "temp", "hum", "windspeed", "cnt_2d_bfr"]
BIKE_KAT = ["season", "holiday", "workday", "weather"]
bike = bike[BIKE_OZ + ["cnt"]].dropna()
b_egitim, b_test = train_test_split(bike, train_size=2 / 3, random_state=42)

bike_rf = Pipeline([
    ("on", ColumnTransformer([("kat", OneHotEncoder(handle_unknown="ignore"), BIKE_KAT)], remainder="passthrough")),
    ("rf", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
]).fit(b_egitim[BIKE_OZ], b_egitim["cnt"])
print(f"Bisiklet RF test R² = {r2_score(b_test['cnt'], bike_rf.predict(b_test[BIKE_OZ])):.3f}")

Xb = b_test[BIKE_OZ].reset_index(drop=True)

# %% [markdown]
# ### 2.1 Doğrulama: kendi ICE kodumuz scikit-learn ile aynı sonucu veriyor mu?

# %%
sk = partial_dependence(bike_rf, Xb, ["temp"], kind="individual", grid_resolution=30)
sk_izgara, sk_ice = sk["grid_values"][0], sk["individual"][0]
bizim_ice = ice_hesapla(bike_rf.predict, Xb, "temp", sk_izgara)
print("Maksimum mutlak fark:", np.abs(bizim_ice - sk_ice).max())
assert np.allclose(bizim_ice, sk_ice), "ICE uygulaması sklearn ile uyuşmuyor!"
print("✔ Sıfırdan yazılan ICE, sklearn.inspection.partial_dependence(kind='individual') ile birebir aynı.")

# %% [markdown]
# ### 2.2 Kitaptaki şekil: sıcaklık, nem ve rüzgâr hızı için ICE
#
# Grafikler kitabın kullandığı R paketi `iml`'in görünümüyle çizilmiştir: test verisinin aralığında **20 noktalı** eşit aralıklı
# ızgara, yarı saydam siyah ICE çizgileri, altta özniteliğin veri dağılımını gösteren **rug** çizgileri ve ggplot2 `theme_bw`.
#
# > *Şekil (kitap, `fig-ice-bike`): Sıcaklık, nem ve rüzgâr hızına göre tahmin edilen bisiklet kiralamalarının ICE grafikleri.*

# %%
fig, eksenler = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
for ax, (oz, ad) in zip(eksenler, [("temp", "Sıcaklık"), ("hum", "Bağıl nem"), ("windspeed", "Rüzgâr hızı")]):
    iz = izgara_olustur(Xb[oz], IML_IZGARA)
    ice_ciz(ax, iz, ice_hesapla(bike_rf.predict, Xb, oz, iz), pdp=False)
    rug(ax, Xb[oz])
    ax.set_xlabel(ad)
eksenler[0].set_ylabel("Tahmin edilen bisiklet kiralama sayısı")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 Kitaptaki şekil: mevsime göre renklendirilmiş nem ICE'si
#
# > *Şekil (kitap, `fig-ice-bike-colored`): Bisiklet kiralamalarını tahmin eden rastgele orman için ICE eğrileri. Çizgiler mevsime göre renklendirilmiştir.*

# %%
iz = izgara_olustur(Xb["hum"], IML_IZGARA)
ice_hum = ice_hesapla(bike_rf.predict, Xb, "hum", iz)
mevsim_tr = Xb["season"].map({"WINTER": "Kış", "SPRING": "İlkbahar", "SUMMER": "Yaz", "FALL": "Sonbahar"}).to_numpy()
renk = dict(zip(["Kış", "İlkbahar", "Yaz", "Sonbahar"], VIRIDIS_4))
fig, ax = plt.subplots(figsize=(8.5, 6))
ice_ciz(ax, iz, ice_hum, alfa=0.3, pdp=False, etiketler=mevsim_tr, renkler=renk)
ax.set(xlabel="Bağıl nem", ylabel="Tahmin edilen bisiklet kiralama sayısı", ylim=(0, None))
efsane_sag(ax, "Mevsim")
plt.show()

# %% [markdown]
# ### 2.4 Kitaptaki şekil: penguenlerde gaga uzunluğu ICE'si
#
# Çeviride not düştüğüm gibi kitaptaki grafik olasılık değil **sınıf etiketi (0/1)** çizer; bu yüzden "çirkin" ve sıçramalıdır.
# Önce kitaptaki hali, ardından (kitabın ötesi) olasılık halini çiziyorum.
#
# > *Şekil (kitap, `fig-ice-penguins`): Gaga uzunluğuna göre dişi sınıfı göstergesinin ICE grafiği (0: erkek, 1: dişi). Her çizgi bir pengueni temsil eder.*

# %%
peng = veri_oku("penguins.csv").dropna()
P_OZ = ["species", "bill_depth_mm", "bill_length_mm", "flipper_length_mm", "body_mass_g"]
p_egitim, p_test = train_test_split(peng, train_size=2 / 3, random_state=2)
peng_rf = Pipeline([
    ("on", ColumnTransformer([("kat", OneHotEncoder(handle_unknown="ignore"), ["species"])], remainder="passthrough")),
    ("rf", RandomForestClassifier(n_estimators=500, random_state=2, n_jobs=-1)),
]).fit(p_egitim[P_OZ], p_egitim["sex"] == "FEMALE")
Xp = p_test[P_OZ].reset_index(drop=True)
iz = izgara_olustur(Xp["bill_length_mm"], IML_IZGARA)

fig, ax = plt.subplots(figsize=(8.5, 6))
ice_ciz(ax, iz, ice_hesapla(lambda D: peng_rf.predict(D).astype(float), Xp, "bill_length_mm", iz), pdp=False)
rug(ax, Xp["bill_length_mm"])
ax.set(xlabel="Gaga uzunluğu (mm)", ylabel="Tahmin edilen dişi sınıfı (0/1)", yticks=[0, 0.25, 0.5, 0.75, 1])
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
plt.show()

# %% [markdown]
# **Kitabın ötesi:** aynı model için sınıf yerine olasılık, P(dişi), çizilince eğriler yumuşar ve okunur hale gelir.

# %%
fig, ax = plt.subplots(figsize=(8.5, 5))
ice_ciz(ax, iz, ice_hesapla(lambda D: peng_rf.predict_proba(D)[:, 1], Xp, "bill_length_mm", iz), pdp=False)
rug(ax, Xp["bill_length_mm"])
ax.set(xlabel="Gaga uzunluğu (mm)", ylabel="Tahmin edilen P(dişi)", ylim=(-0.03, 1.03))
plt.show()

# %% [markdown]
# ### 2.5 Kitaptaki şekil: merkezlenmiş ICE (c-ICE)
#
# Kitaptaki gibi referans noktası **tüm verideki en düşük sıcaklıktır** (`center.at = min(bike$temp)`); bu nokta ızgaraya eklenir.
#
# > *Şekil (kitap, `fig-ice-bike-centered`): Sıcaklığa göre tahmin edilen bisiklet sayısı için merkezlenmiş ICE grafikleri. Çizgiler, sıcaklık gözlenen minimumunda sabitken elde edilen tahmine göre farkı gösterir.*

# %%
a_ref = bike["temp"].min()
iz = np.unique(np.r_[a_ref, izgara_olustur(Xb["temp"], IML_IZGARA)])
ice_t = ice_hesapla(bike_rf.predict, Xb, "temp", iz)
fig, ax = plt.subplots(figsize=(8.5, 6))
ice_ciz(ax, iz, merkezle(ice_t), pdp=False)
rug(ax, Xb["temp"])
ax.set(xlabel="Sıcaklık", ylabel=f"Sıcaklık = {a_ref:.1f} noktasındaki tahmine göre fark")
plt.show()

# %% [markdown]
# ### 2.6 Kitabın ötesi: türev ICE (d-ICE)
#
# Kitap d-ICE'yi anlatır ama şeklini vermez. Aynı modelden sonlu farklarla hesaplıyorum; mavi çizgi türevlerin standart sapmasıdır
# (yüksek olduğu yerde örnekler arasında heterojenlik, yani etkileşim vardır).

# %%
iz = izgara_olustur(Xb["temp"], 40)
d_ice = turev(ice_hesapla(bike_rf.predict, Xb, "temp", iz), iz)
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(iz, d_ice.T, color="black", alpha=0.15, lw=0.6)
ax.plot(iz, d_ice.std(axis=0), color="#1F77B4", lw=2.5, label="Türevlerin std. sapması")
rug(ax, Xb["temp"])
ax.set(xlabel="Sıcaklık", ylabel="∂f̂ / ∂sıcaklık")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 3. Asıl uygulama: California Housing
#
# **Veri:** 1990 ABD nüfus sayımından 20.640 California bölgesi (Pace & Barry, 1997, *Statistics & Probability Letters* 33(3)).
# Hedef: bölgedeki medyan ev değeri. Öznitelikler: konum (enlem, boylam), bina yaşı, toplam oda, toplam yatak odası, nüfus,
# hane sayısı, medyan gelir ve okyanusa yakınlık.
#
# **Bu veri neden zorlu?**
#
# | Sorun | ICE açısından anlamı |
# |---|---|
# | `total_rooms`, `total_bedrooms`, `population`, `households` arasında 0.85–0.98 korelasyon | Birini değiştirip diğerlerini sabit tutmak eğitim verisinden uzak öznitelik birleşimleri üretebilir |
# | Enlem–boylam birlikte California'nın şeklini çizer | Enlemi değiştirmek bir bölgeyi okyanusa ya da Nevada'ya taşır |
# | Hedef 500.001 $'da kırpılmış (verinin ~%5'i) | Kırpma, model tahminlerinin üst gelir aralığındaki davranışını etkileyebilir |
# | `total_bedrooms`'ta 207 eksik değer, sağa çarpık sayım değişkenleri | Eşit aralıklı ızgaranın çoğu boş bölgeye düşer |
# | Coğrafi heterojenlik (kıyı / iç bölge / ada) | Ortalama etki (PDP) bireysel etkiyi temsil etmeyebilir: ICE'nin asıl gücü burada |

# %%
ev = veri_oku("housing.csv")
ev["median_house_value"] = ev["median_house_value"] / 100_000    # hedef: 100 bin $ biriminde
SAYISAL = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
           "population", "households", "median_income"]
OZ = SAYISAL + ["ocean_proximity"]
print(ev.shape)
print("Kırpılmış hedef oranı:", f"{(ev['median_house_value'] >= 5.00001).mean():.1%}")
print(ev[SAYISAL].corr().round(2).loc[["total_rooms", "total_bedrooms", "population", "households"],
                                      ["total_rooms", "total_bedrooms", "population", "households"]])

# %%
e_egitim, e_test = train_test_split(ev, train_size=2 / 3, random_state=42)
ev_model = Pipeline([
    ("on", ColumnTransformer([("kat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"])],
                             remainder="passthrough", sparse_threshold=0)),
    ("gb", HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=31, random_state=42)),
]).fit(e_egitim[OZ], e_egitim["median_house_value"])
tahmin = ev_model.predict(e_test[OZ])
print(f"Gradient boosting  test R² = {r2_score(e_test['median_house_value'], tahmin):.3f}, "
      f"MAE = {mean_absolute_error(e_test['median_house_value'], tahmin) * 100_000:,.0f} $")

# ICE grafikleri okunabilir kalsın diye test kümesinden 400 bölgelik bir örneklem
Xe = e_test[OZ].sample(400, random_state=1).reset_index(drop=True)
yakinlik = Xe["ocean_proximity"].to_numpy()
RENK_OKYANUS = {"INLAND": "#e66101", "<1H OCEAN": "#5e3c99", "NEAR OCEAN": "#1b9e77", "NEAR BAY": "#0571b0", "ISLAND": "#000000"}

# %% [markdown]
# ### 3.1 PDP neyi gizliyor? Medyan gelirin ICE'si
#
# Izgara medyan gelirin %1–%99 kantilleri arasında kurulmuştur (gelir 15'te kırpıldığı için uç değerler anlamsızdır).

# %%
iz_gelir = izgara_olustur(e_egitim["median_income"], 50, 0.01, 0.99)
ice_gelir = ice_hesapla(ev_model.predict, Xe, "median_income", iz_gelir)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
ice_ciz(a1, iz_gelir, ice_gelir, alfa=0.12)
rug(a1, Xe["median_income"]); rug(a2, Xe["median_income"])
a1.set(xlabel="Medyan gelir (10 bin $)", ylabel="Tahmin edilen ev değeri (100 bin $)", title="ICE + PDP")
a1.legend()
ice_ciz(a2, iz_gelir, ice_gelir, alfa=0.25, pdp=False, etiketler=yakinlik,
        renkler={k: v for k, v in RENK_OKYANUS.items() if k in set(yakinlik)})
a2.set(xlabel="Medyan gelir (10 bin $)", title="Okyanusa yakınlığa göre renklendirilmiş ICE")
a2.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# Sağdaki grafikte iç bölgelerin (`INLAND`, turuncu) eğrileri kıyı bölgelerinden **daha aşağıda** başlar. Ancak bu yalnızca bir
# **seviye (kesişim) farkıdır**: eğriler birbirine paralel ilerliyor gibi görünür. Seviye farkını eğim farkından ayırmanın yolu
# c-ICE'dir. Bütün eğrileri sol uçta sıfıra sabitleyip "gelir arttıkça tahmin ne kadar artıyor?" sorusuna bakıyoruz.

# %%
c_gelir = merkezle(ice_gelir)
d_gelir = turev(ice_gelir, iz_gelir)
yas_grubu = pd.cut(Xe["housing_median_age"], [0, 20, 35, 52], labels=["Yeni (≤20 yıl)", "Orta (21–35)", "Eski (>35 yıl)"]).astype(str).to_numpy()
RENK_YAS = {"Yeni (≤20 yıl)": "#1b9e77", "Orta (21–35)": "#7570b3", "Eski (>35 yıl)": "#d95f02"}
orta = len(iz_gelir) // 2

print("c-ICE değerinin (gelirin etkisi) gruplara göre ortalaması, 100 bin $:")
for ad, etiket in [("Okyanusa yakınlık", yakinlik), ("Bina yaşı", yas_grubu)]:
    tablo = pd.DataFrame({f"gelir {iz_gelir[0]:.1f}→{iz_gelir[orta]:.1f}": c_gelir[:, orta],
                          f"gelir {iz_gelir[0]:.1f}→{iz_gelir[-1]:.1f}": c_gelir[:, -1]}).groupby(etiket).mean()
    print(f"\n{ad}:\n{tablo.round(2)}")
rho = pd.Series(c_gelir[:, orta]).corr(Xe["housing_median_age"], method="spearman")
print(f"\nOrta noktadaki gelir etkisi ile bina yaşı arasındaki Spearman korelasyonu: {rho:.2f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
ice_ciz(a1, iz_gelir, c_gelir, alfa=0.25, pdp=False, etiketler=yas_grubu, renkler=RENK_YAS)
a1.plot(iz_gelir, c_gelir.mean(0), color="#d62728", lw=3, label="c-PDP")
rug(a1, Xe["median_income"])
a1.set(xlabel="Medyan gelir (10 bin $)", ylabel=f"gelir = {iz_gelir[0]:.2f} noktasına göre fark (100 bin $)", title="c-ICE, bina yaşına göre renkli")
a1.legend(fontsize=8)
a2.plot(iz_gelir, d_gelir.T, color="0.3", alpha=0.07, lw=0.8)
a2.plot(iz_gelir, d_gelir.mean(0), color="#d62728", lw=2.5, label="ortalama türev")
a2.plot(iz_gelir, d_gelir.std(0), color="#1f77b4", lw=2.5, label="türevlerin std. sapması")
a2.set(xlabel="Medyan gelir (10 bin $)", ylabel="∂f̂ / ∂gelir", title="d-ICE", ylim=(np.percentile(d_gelir, 1), np.percentile(d_gelir, 99)))
a2.legend(fontsize=8)
fig.tight_layout()
plt.show()


# %% [markdown]
# **Bulgu (tabloya bakarak):**
#
# * Alt ve üst gelir noktaları arasındaki tahmin artışının grup ortalamaları birbirine yakın (~270 bin $).
#   Tablonun son sütunu, her gruptaki bölgelerin tahmin farklarını ortalıyor.
# * Fark eğrinin **ortasında** ortaya çıkar. Gelir orta düzeye çıktığında **eski** mahallelerde tahmin yeni mahallelere
#   göre belirgin biçimde daha fazla yükselir. Bina yaşı ile orta noktadaki etki arasındaki Spearman korelasyonu yaklaşık 0,6'dır.
#   Eski mahallelerin eğrileri erken ve dik yükselip sonra düzleşir, yeni mahallelerinki ise geç yükselir.
# * Okyanusa yakınlık da benzer ama daha zayıf bir fark yaratır (kıyı bölgeleri orta noktada iç bölgelerden biraz önde).
#   Okyanusun asıl etkisi önceki grafikteki **seviye kaymasıdır**.
#
# Bu desen modelde **gelir × bina yaşı etkileşimine** işaret ediyor. Uçtan uca ortalama artışlar yakın olsa da tahminin
# hangi gelir aralığında daha hızlı yükseldiği bina yaşı gruplarına göre değişiyor.
# PDP'nin tek kırmızı çizgisi bu iki farklı şekli ortalayarak kaybeder. d-ICE'deki türevlerin standart sapması da
# heterojenliğin hangi gelir aralığında yoğunlaştığını gösterir.
#
# *Not:* d-ICE grafiği dalgalıdır, çünkü gradient boosting ağaçlardan oluşur ve tahmin işlevi **basamaklıdır**. Türev çoğu yerde
# sıfır, basamaklarda ise çok büyüktür. Kitabın d-ICE için söylediği "oldukça pratik değildir" uyarısı ağaç modellerinde
# özellikle geçerlidir; ortalama ve standart sapma çizgileri ham eğrilerden daha bilgilendiricidir.

# %% [markdown]
# ### 3.2 Kitabın ötesi: ICE eğrilerini kümelemek
#
# 400 çizgiye gözle bakmak zordur. c-ICE eğrilerini bir **vektör** olarak düşünüp k-ortalamalar ile kümelersek,
# modelin gelir için öğrendiği farklı "tepki tiplerini" ve bu tiplerin hangi bölgelerden geldiğini görebiliriz.
# Burada etkileşimi önceden bilmemiz gerekmez; kümeler veriden çıkar ve sonra hangi özniteliklerle ilişkili olduklarına bakarız.

# %%
km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(c_gelir)
sira = np.argsort([c_gelir[km.labels_ == k, -1].mean() for k in range(3)])
kume = np.empty_like(km.labels_)
for yeni, eski in enumerate(sira):
    kume[km.labels_ == eski] = yeni
kume_ad = np.array(["Zayıf tepki", "Orta tepki", "Güçlü tepki"])[kume]
renk_k = {"Zayıf tepki": "#1b9e77", "Orta tepki": "#7570b3", "Güçlü tepki": "#d95f02"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
ice_ciz(a1, iz_gelir, c_gelir, alfa=0.15, pdp=False, etiketler=kume_ad, renkler=renk_k)
for ad, r in renk_k.items():
    a1.plot(iz_gelir, c_gelir[kume_ad == ad].mean(0), color=r, lw=3.5)
a1.set(xlabel="Medyan gelir (10 bin $)", ylabel="c-ICE (100 bin $)", title="c-ICE kümeleri (k = 3)")
a1.legend(fontsize=8)
a2.scatter(e_egitim["longitude"], e_egitim["latitude"], s=1, color="0.85")
for ad, r in renk_k.items():
    m = kume_ad == ad
    a2.scatter(Xe.loc[m, "longitude"], Xe.loc[m, "latitude"], s=12, color=r, label=ad)
a2.set(xlabel="Boylam", ylabel="Enlem", title="Kümeler haritada")
a2.legend(fontsize=8)
fig.tight_layout()
plt.show()

ozet = Xe.assign(küme=kume_ad).groupby("küme")[["housing_median_age", "median_income", "longitude", "latitude"]].median()
ozet.insert(0, "bölge sayısı", pd.Series(kume_ad).value_counts())
print(ozet.loc[["Zayıf tepki", "Orta tepki", "Güçlü tepki"]].round(2))
print("\nKümelerin bina yaşı gruplarına dağılımı (satır yüzdesi):")
print(pd.crosstab(kume_ad, yas_grubu, normalize="index").round(2))

# %% [markdown]
# Gözetimsiz kümeleme aynı sonuca ulaşır: "güçlü tepki" kümesi ağırlıklı olarak **eski** mahallelerden oluşur ve
# Los Angeles ve San Francisco Körfezi'nin yerleşik merkezlerinde yoğunlaşır. "Zayıf tepki" kümesi ise ağırlıklı olarak
# yeni yerleşimlerdir ve Central Valley gibi iç bölgelere dağılmıştır.

# %% [markdown]
# ### 3.3 Sınırlılık: ICE’nin ürettiği, eğitim verisinden uzak noktalar
#
# Kitabın *Sınırlılıklar* bölümü: *"İlgilenilen öznitelik diğer özniteliklerle ilişkiliyse, çizgilerdeki bazı noktalar
# ortak öznitelik dağılımına göre geçersiz veri noktaları olabilir."* Bunu **ölçmek** için basit bir gerçekçilik testi kuruyorum:
#
# 1. Sayısal öznitelikleri (sayım değişkenlerini log ölçeğinde) standartlaştır.
# 2. Eğitim verisinde her noktanın **en yakın komşusuna uzaklığını** hesapla; bu uzaklıkların %99 kantili eşik $\tau$ olsun.
# 3. ICE ızgarasındaki her yapay nokta için en yakın gerçek bölgeye uzaklığı hesapla. Uzaklık $> \tau$ ise nokta
#    kullanılan uzaklık ölçütüne göre **veri dışı (OOD)** işaretlenir. Bu işaret eğitim verisinin desteğini ölçer.
# 4. Aynı testi yalnızca (boylam, enlem) için de yap: haritada hiçbir gerçek bölgeye yakın olmayan nokta da OOD'dir.

# %%
LOG_OZ = ["total_rooms", "total_bedrooms", "population", "households"]


def gerceklik_uzayi(D):
    Z = D[SAYISAL].copy()
    Z[LOG_OZ] = np.log1p(Z[LOG_OZ])
    return Z.fillna(Z.median())


olcek = StandardScaler().fit(gerceklik_uzayi(e_egitim))
referans = olcek.transform(gerceklik_uzayi(e_egitim))
nn = NearestNeighbors(n_neighbors=2).fit(referans)
tau = np.quantile(nn.kneighbors(referans)[0][:, 1], 0.99)   # 1. komşu kendisi olduğu için 2.'yi alıyoruz

# Coğrafi kontrol: 8 boyutlu uzaklıkta enlem tek başına küçük bir pay alır; bu yüzden ayrıca haritada
# en yakın gerçek bölgeye uzaklığa bakıyoruz (okyanusa ya da Nevada'ya düşen noktaları yakalamak için).
konum = e_egitim[["longitude", "latitude"]].to_numpy()
nn_geo = NearestNeighbors(n_neighbors=2).fit(konum)
tau_geo = np.quantile(nn_geo.kneighbors(konum)[0][:, 1], 0.99)
print(f"Gerçekçilik eşikleri: τ (öznitelik uzayı) = {tau:.3f}, τ_geo (harita, derece) = {tau_geo:.3f}")


def ood_matrisi(X, ozellik, izgara):
    """ICE ızgarasındaki her (örnek, z) noktası için 'veri dışı mı?' maskesi (n x K)."""
    n, K = len(X), len(izgara)
    X_cogul = pd.concat([X] * K, ignore_index=True)
    X_cogul[ozellik] = np.repeat(izgara, n)
    uzaklik = nn.kneighbors(olcek.transform(gerceklik_uzayi(X_cogul)), n_neighbors=1)[0][:, 0]
    uzaklik_geo = nn_geo.kneighbors(X_cogul[["longitude", "latitude"]].to_numpy(), n_neighbors=1)[0][:, 0]
    return ((uzaklik > tau) | (uzaklik_geo > tau_geo)).reshape(K, n).T


sonuc = []
for oz in ["median_income", "housing_median_age", "total_rooms", "population", "latitude"]:
    iz = izgara_olustur(e_egitim[oz], 30)                         # kitaptaki gibi min–max ızgara
    sonuc.append({"öznitelik": oz, "ICE noktalarının OOD oranı": ood_matrisi(Xe, oz, iz).mean()})
ood_tablo = pd.DataFrame(sonuc).set_index("öznitelik").sort_values("ICE noktalarının OOD oranı")
print((ood_tablo * 100).round(1).rename(columns=lambda c: c + " (%)"))

# %% [markdown]
# `total_rooms` ve `population` için ICE noktalarının büyük kısmı seçilen uzaklık eşiğini aşıyor.
# Eğitim verisindeki desteğin zayıf olduğu kısımları göstermek için bu noktaları grafikte işaretliyoruz.

# %%
def ice_gerceklik_ciz(ax, X, ozellik, izgara, baslik, xlog=False):
    ice = ice_hesapla(ev_model.predict, X, ozellik, izgara)
    ood = ood_matrisi(X, ozellik, izgara)
    for i in range(len(X)):
        y_ic = np.where(~ood[i], ice[i], np.nan)
        ax.plot(izgara, ice[i], color="#d62728", alpha=0.18, lw=0.8)
        ax.plot(izgara, y_ic, color="#1f77b4", alpha=0.35, lw=0.9)
    ax.plot([], [], color="#1f77b4", lw=2, label="gerçekçi bölge")
    ax.plot([], [], color="#d62728", lw=2, label="veri dışı (OOD)")
    ax2 = ax.twinx()
    ax2.fill_between(izgara, ood.mean(0), color="0.5", alpha=0.15, step="mid")
    ax2.set_ylim(0, 1); ax2.set_ylabel("OOD oranı", color="0.4"); ax2.grid(False, which="both")
    ax.set(title=baslik, xlabel=ozellik, ylabel="Tahmin (100 bin $)")
    if xlog:
        ax.set_xscale("log")
    ax.legend(fontsize=8, loc="upper left")
    return ice, ood


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
ice_gerceklik_ciz(a1, Xe.head(150), "median_income", izgara_olustur(e_egitim["median_income"], 40), "Medyan gelir: çoğu nokta gerçekçi")
ice_gerceklik_ciz(a2, Xe.head(150), "total_rooms", np.geomspace(10, e_egitim["total_rooms"].max(), 40),
                  "Toplam oda sayısı: veri desteği düşük", xlog=True)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4 Zafiyeti gözle görmek: enlemi değiştirince bölge nereye gidiyor?
#
# Los Angeles'tan tek bir bölgeyi alıp yalnızca **enlemini** değiştiriyoruz (boylam sabit, *ceteris paribus*).
# ICE bu bölgeyi güneyde okyanusun ortasına, kuzeyde California'nın dışına (Nevada) taşır; model ise oralarda da
# hiçbir uyarı vermeden bir tahmin üretir.

# %%
la = Xe.iloc[[np.argmin((Xe["longitude"] + 118.25) ** 2 + (Xe["latitude"] - 34.05) ** 2)]].reset_index(drop=True)
iz_enlem = izgara_olustur(e_egitim["latitude"], 60)
ice_la = ice_hesapla(ev_model.predict, la, "latitude", iz_enlem)[0]
ood_la = ood_matrisi(la, "latitude", iz_enlem)[0]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.2]})
a1.scatter(e_egitim["longitude"], e_egitim["latitude"], s=1, color="0.8")
a1.scatter(np.full(60, la["longitude"][0]), iz_enlem, c=np.where(ood_la, "#d62728", "#1f77b4"), s=18)
a1.scatter(la["longitude"], la["latitude"], s=150, marker="*", color="gold", edgecolor="k", zorder=5, label="gerçek bölge (LA)")
a1.set(xlabel="Boylam", ylabel="Enlem", title="ICE'nin oluşturduğu yapay bölgeler")
a1.legend(fontsize=8)
a2.plot(iz_enlem, ice_la, color="0.6", lw=1)
a2.scatter(iz_enlem, ice_la, c=np.where(ood_la, "#d62728", "#1f77b4"), s=18)
a2.axvline(la["latitude"][0], color="gold", lw=2, ls="--")
a2.set(xlabel="Enlem", ylabel="Tahmin (100 bin $)", title="Aynı bölgenin ICE eğrisi (kırmızı = veri dışı)")
fig.tight_layout()
plt.show()
print(f"Bu tek eğrideki yapay noktaların {ood_la.mean():.0%}'i veri dağılımının dışında.")

# %% [markdown]
# ### 3.5 Kıyı bölgelerinde gelir ICE’leri ve hedefin üst sınırı
#
# Bazı ICE eğrileri yüksek gelir değerlerinde düzleşiyor. Kırpma sınırı hedef değerlerine uygulandığı için
# modelin tahminleri bu sınırı aşabilir. Örneğin bir çalıştırmada en yüksek tahmin 5,40 (yaklaşık 540.000 $).
# Düzleşmenin kırpmayla ilişkisini sınamak için kırpılmamış hedefle bir karşılaştırma gerekir.

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
zengin = Xe["ocean_proximity"].isin(["NEAR BAY", "<1H OCEAN"]).to_numpy()
ax.plot(iz_gelir, ice_gelir[zengin].T, color="#5e3c99", alpha=0.12)
ax.axhline(5.00001, color="k", ls="--", lw=1.5, label="hedefin kırpıldığı değer (500.001 $)")
ax.set(xlabel="Medyan gelir (10 bin $)", ylabel="Tahmin (100 bin $)", title="Kıyı bölgeleri: gelir ICE’leri ve hedef sınırı")
ax.legend()
plt.show()
print(f"Kıyı bölgelerinde en yüksek ICE değeri: {ice_gelir[zengin].max():.2f} (kırpma: 5.00)")

# %% [markdown]
# ## 4. Özet
#
# | Bulgu | Kanıt |
# |---|---|
# | Sıfırdan yazılan ICE doğru | `sklearn` ile farkı 0 (Bölüm 2.1) |
# | Kitaptaki şekiller Python'da yeniden üretildi | Bölüm 2.2 – 2.5 |
# | Okyanusa yakınlık ağırlıklı olarak seviyeyi kaydırıyor | İç bölgelerin ICE'leri aşağıda başlıyor; uçtan uca tahmin artışının grup ortalamaları birbirine yakın (Bölüm 3.1) |
# | PDP, gelir × bina yaşı etkileşimini gizliyor | c-ICE: eski mahallelerde tahmin orta gelir düzeyinde çok daha erken yükseliyor (Spearman ≈ 0,6); gözetimsiz kümeleme aynı sonucu buluyor (Bölüm 3.1 – 3.2) |
# | d-ICE'deki türev std. sapması etkileşimin nerede yoğunlaştığını gösteriyor | Bölüm 3.1 |
# | Korelasyonlu özniteliklerde ICE eğitim verisinden uzak noktalar üretiyor | `total_rooms` ve `population` ICE noktalarının yaklaşık %80'i veri dışında (Bölüm 3.3) |
# | Coğrafi öznitelikte ICE bölgeyi haritadan çıkarıyor | Enlem ICE noktalarının ~%65'i veri dışında; LA bölgesi okyanusa ve Nevada'ya taşınıyor (Bölüm 3.3 – 3.4) |
# | Bazı ICE eğrileri yüksek gelirlerde düzleşiyor, kimi tahminler hedefin kırpma sınırını aşıyor | Bölüm 3.5 |
#
# **Mühendislik dersi:** ICE, *"bu örnek için model ne yapıyor?"* sorusuna dürüst bir cevap verir; ama *"gerçekte ne olur?"*
# sorusuna cevap vermez. Korelasyonlu verilerde ICE'yi her zaman bir **gerçekçilik kontrolü** (burada en yakın komşu testi) ile
# birlikte raporlamak ve veri dışındaki parçaları soluk göstermek gerekir. Alternatif olarak ALE grafikleri
# (koşullu dağılım üzerinden çalışır) bu sorundan daha az etkilenir.
