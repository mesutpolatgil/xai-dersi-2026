# %% [markdown]
# # Bölüm 8 – GLM, GAM ve ötesi: Uygulama
#
# Ahmet Batuhan Kaya – Açıklanabilir Yapay Zeka (XAI), 2026 Güz
#
# Kitap, doğrusal modelin üç sorununu ve çözümlerini anlatıyor:
#   Sorun 1: Hedef Gauss dağılımlı değil      → GLM (bağlantı fonksiyonu + uygun dağılım)
#   Sorun 2: Öznitelikler etkileşimde         → Etkileşim terimleri
#   Sorun 3: İlişki doğrusal değil            → Dönüşüm, kategorize etme, GAM
#
# Akış:
#   0. Kurulum
#   1. Köprü: Kitaptaki kahve simülasyonu (doğrusal model vs Poisson GLM) + güven aralığı kapsama deneyi
#   2. Köprü: Bisiklet verisinde sıcaklık × iş günü etkileşimi
#   3. Köprü: Bisiklet verisinde sıcaklığın doğrusal olmayan etkisi (4 yöntem)
#   4. Zorlu veri: freMTPL2 – Fransız araç sigortası, 678.000 poliçe
#   5. Poisson GLM: offset (maruz kalma süresi) ve çarpımsal yorum
#   6. Kırma 1: Aşırı yayılım (overdispersion) – güven aralıkları neden yalan söyler?
#   7. Sürücü yaşının doğrusal olmayan etkisi: doğrusal vs kategorize vs GAM
#   8. Tuzak: "y'yi log'lamak" ≠ "log bağlantı fonksiyonu"
#   9. Kırma 2: Birbirinin kopyası öznitelikler (Area ve Density)
#
# VS Code'da her "# %%" satırının üstünde "Run Cell" yazar; hücre hücre çalıştırın.

# %% 0. Kurulum
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import LinearGAM, PoissonGAM, s, l

warnings.filterwarnings("ignore", category=FutureWarning)
RNG = np.random.default_rng(42)
SEKIL_KLASORU = "sekiller"
os.makedirs(SEKIL_KLASORU, exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 10)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(os.path.join(SEKIL_KLASORU, ad), dpi=150, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 1. Köprü: Kitaptaki kahve simülasyonu
#
# Kitap 200 gün için kahve sayısı üretiyor. Gerçek ilişki (kitabın kodundan):
#   ln(λ) = 0,1·stres − 0,2·(uyku − 5) − 1·(iş yok)
#   kahve ~ Poisson(λ)
# Sayım verisi: negatif olamaz, tam sayıdır, varyansı ortalamasıyla birlikte büyür.
# GLM'in üç bileşeni:
#   (1) Dağılım: Poisson   (2) Doğrusal yordayıcı: β0 + β1·x1 + ...   (3) Bağlantı: log
#   ln(E[y]) = Xβ   ⇔   E[y] = exp(Xβ)   → tahmin her zaman pozitif.


def kahve_uret(n=200):
    d = pd.DataFrame({"stres": RNG.uniform(1, 10, n), "uyku": RNG.uniform(1, 10, n),
                      "is_gunu": RNG.choice(["EVET", "HAYIR"], n)})
    lam = np.exp(1 * d.stres / 10 - 2 * (d.uyku - 5) / 10 - 1 * (d.is_gunu == "HAYIR"))
    d["kahve"] = RNG.poisson(lam)
    return d


# %% 1a. Doğrusal model neden başarısız?
kahve = kahve_uret()
formul_k = "kahve ~ stres + uyku + C(is_gunu, Treatment('HAYIR'))"
lm_k = smf.ols(formul_k, kahve).fit()
glm_k = smf.glm(formul_k, kahve, family=sm.families.Poisson()).fit()

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].hist(kahve.kahve, bins=range(kahve.kahve.max() + 2), edgecolor="white")
axes[0].set(title="Gerçek kahve sayıları", xlabel="Kahve / gün")
axes[1].hist(lm_k.fittedvalues, bins=25, color="tab:red", edgecolor="white")
axes[1].axvline(0, color="black")
axes[1].set(title=f"Doğrusal model tahminleri ({(lm_k.fittedvalues < 0).sum()} negatif!)", xlabel="Tahmin")
axes[2].hist(glm_k.fittedvalues, bins=25, color="tab:green", edgecolor="white")
axes[2].set(title="Poisson GLM tahminleri (hepsi > 0)", xlabel="Tahmin")
kaydet("01_kahve_dogrusal_vs_glm.png")

gercek = pd.Series({"Intercept": 0.0, "C(is_gunu, Treatment('HAYIR'))[T.EVET]": 1.0,
                    "stres": 0.1, "uyku": -0.2})
tablo = pd.DataFrame({"ağırlık": glm_k.params, "exp(ağırlık)": np.exp(glm_k.params),
                      "gerçek ağırlık": gercek}).round(3)
print(tablo)
# Not: Kitaptaki formülde sabit terim 0 + 0,2·5 − 1 = 0 çıkar; uyku ve stres katsayıları 0,1 ve −0,2'dir.
# Yorum (çarpımsal): stres +1 → beklenen kahve exp(β) katına çıkar; iş günü → yaklaşık e ≈ 2,7 katı.

# %% 1b. Güven aralığı ne demek? Kapsama deneyi
# Kitap (Bölüm 6): "Kestirimi 100 kez tekrarlasaydık, %95'lik güven aralığı
# 100 durumun 95'inde gerçek ağırlığı içerirdi." Bunu gerçekten yapalım.
kapsadi_glm = 0
for _ in range(200):
    d = kahve_uret()
    ga = smf.glm(formul_k, d, family=sm.families.Poisson()).fit().conf_int().loc["stres"]
    kapsadi_glm += ga[0] <= 0.1 <= ga[1]
print(f"Poisson GLM: 200 tekrarın %{100 * kapsadi_glm / 200:.1f}'inde %95 GA gerçek değeri (0,1) kapsadı.")

# %% [markdown]
# ## 2. Köprü: Etkileşim (bisiklet verisi)
#
# Kitap: İş günlerinde insanlar hava ne olursa olsun bisikletle işe gider; tatilde ise
# yalnızca hava güzelse binerler. Yani sıcaklığın etkisi iş gününe BAĞLI → etkileşim.
# Modele temp × workday çarpım terimi eklemek, her grup için ayrı eğim demektir.

# %% 2. Bisiklet etkileşim modeli
bike = pd.read_csv("https://raw.githubusercontent.com/christophM/"
                   "interpretable-ml-book/master/data/bike.csv")
etk = smf.ols("cnt ~ C(season, Treatment('WINTER')) + C(holiday) + C(weather, Treatment('GOOD'))"
              " + hum + windspeed + cnt_2d_bfr + temp * C(workday, Treatment('N'))", bike).fit()
b_temp = etk.params["temp"]
b_etk = etk.params["temp:C(workday, Treatment('N'))[T.Y]"]
print(f"Sıcaklık eğimi – iş günü DEĞİL: {b_temp:.1f}")
print(f"Etkileşim katsayısı:           {b_etk:.1f}")
print(f"Sıcaklık eğimi – iş günü:      {b_temp:.1f} + ({b_etk:.1f}) = {b_temp + b_etk:.1f}")
print("→ Etkileşim katsayısı negatif ama iş günündeki sıcaklık etkisi hâlâ POZİTİF.")
print("  Etkileşim katsayısı tek başına yorumlanmaz; ana etkiyle toplanır (kitaptaki uyarı).")

fig, ax = plt.subplots(figsize=(7, 4))
for grup, renk in [("N", "tab:orange"), ("Y", "tab:blue")]:
    alt = bike[bike.workday == grup]
    ax.scatter(alt.temp, alt.cnt, s=6, alpha=0.4, color=renk)
    izgara = np.linspace(bike.temp.min(), bike.temp.max(), 50)
    sabit = {"season": "SUMMER", "holiday": "N", "weather": "GOOD", "hum": bike.hum.mean(),
             "windspeed": bike.windspeed.mean(), "cnt_2d_bfr": bike.cnt_2d_bfr.mean(), "workday": grup}
    tahmin = etk.predict(pd.DataFrame({**sabit, "temp": izgara}))
    ax.plot(izgara, tahmin, color=renk, lw=2.5, label=f"İş günü = {grup}")
ax.set(title="Etkileşim: iki grup, iki farklı eğim", xlabel="Sıcaklık (°C)", ylabel="Kiralanan bisiklet")
ax.legend()
kaydet("02_bisiklet_etkilesim.png")

# %% [markdown]
# ## 3. Köprü: Doğrusal olmayan etki – kitaptaki 4 panelli şekil
#
# Sıcaklık arttıkça kiralama artar, ama çok sıcakta düşer. Dört yaklaşım:
#   a) Doğrusal: tek eğim → ters-U'yu göremez
#   b) Dönüşüm (log/kök): biçimi ancak kısmen esnetir
#   c) Kategorize etme: basamak fonksiyonu, kesim noktaları keyfi
#   d) GAM: f(sıcaklık) eğrisini spline'larla veriden öğrenir; düzgünlüğü ceza (λ) ile ayarlanır

# %% 3. Dört model
x = bike[["temp"]].values
y = bike.cnt.values
izgara = np.linspace(x.min(), x.max(), 200)
m_lin = np.polyfit(x[:, 0], y, 1)
m_kok = np.polyfit(np.sqrt(x[:, 0] - x.min() + 1), y, 1)
kesim = pd.cut(bike.temp, bins=8)
ortalama = bike.groupby(kesim, observed=True).cnt.mean()
gam = LinearGAM(s(0, n_splines=10)).gridsearch(x, y, progress=False)   # λ çapraz doğrulama benzeri GCV ile seçilir

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
paneller = [
    ("Doğrusal", np.polyval(m_lin, izgara)),
    ("Karekök dönüşümü", np.polyval(m_kok, np.sqrt(izgara - x.min() + 1))),
    ("Kategorize (8 aralık)", ortalama.reindex(pd.cut(izgara, bins=kesim.cat.categories)).values),
    (f"GAM (spline, λ={gam.lam[0][0]:.1f})", gam.predict(izgara.reshape(-1, 1))),
]
for ax, (ad, egri) in zip(axes.flat, paneller):
    ax.scatter(x, y, s=5, alpha=0.3, color="gray")
    ax.plot(izgara, egri, lw=2.5, color="tab:blue")
    ax.set_title(ad)
for ax in axes[1]:
    ax.set_xlabel("Sıcaklık (°C)")
for ax in axes[:, 0]:
    ax.set_ylabel("Kiralanan bisiklet")
kaydet("03_bisiklet_dogrusal_olmayan.png")

# %% [markdown]
# ## 4. Zorlu veri: freMTPL2 (Fransız araç sigortası)
#
# Sigortacılıkta Poisson GLM'in "ders kitabı" veri seti (Noll, Salzmann & Wüthrich, 2020).
# Her satır bir poliçe. Hedef: yıl içindeki hasar SAYISI (ClaimNb).
# Neden zorlu?
#   - 678.000 satır; poliçelerin ~%95'inde hiç hasar yok (sıfır yığılması).
#   - Poliçeler farklı sürelerle gözlenmiş (Exposure: 0-1 yıl). 3 aylık poliçeyle 1 yıllık
#     poliçenin hasar sayısını doğrudan karşılaştıramayız → offset gerekir.
#   - Sürücü yaşının etkisi U şeklinde (genç ve çok yaşlı sürücü riskli).
#   - Area ve Density neredeyse aynı bilgiyi taşıyor.

# %% 4. Veriyi yükle
def fremtpl_yukle():
    """Önce yerel CSV'ye bakar, yoksa OpenML'den indirir (ilk seferde ~1 dk)."""
    if os.path.exists("freMTPL2freq.csv"):
        d = pd.read_csv("freMTPL2freq.csv")
    else:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
        d.to_csv("freMTPL2freq.csv", index=False)   # bir sonraki çalıştırma için
    for c in ["Area", "VehBrand", "VehGas", "Region"]:
        d[c] = d[c].astype(str).str.strip("'")
    return d


sig = fremtpl_yukle()
# Standart temizlik (scikit-learn'ün resmi örneğindeki gibi): uç değerleri kırp
sig["ClaimNb"] = sig["ClaimNb"].clip(upper=4)
sig["Exposure"] = sig["Exposure"].clip(upper=1)
sig["logDensity"] = np.log(sig["Density"])
sig["siklik"] = sig["ClaimNb"] / sig["Exposure"]            # yıllık hasar sıklığı

print(sig.shape)
print(sig[["ClaimNb", "Exposure", "DrivAge", "BonusMalus", "Density"]].describe().round(2))
print(f"\nHasarsız poliçe oranı: %{100 * (sig.ClaimNb == 0).mean():.1f}")
print(f"Portföy hasar sıklığı: {sig.ClaimNb.sum() / sig.Exposure.sum():.4f} hasar / poliçe-yıl")

# %% [markdown]
# ## 5. Poisson GLM ve offset
#
# Beklenen hasar sayısı = sıklık × süre:
#   E[ClaimNb] = Exposure · exp(Xβ)
#   ln E[ClaimNb] = ln(Exposure) + Xβ
# ln(Exposure) katsayısı 1'e SABİTLENMİŞ bir terim olarak eklenir; buna offset denir.
# Böylece katsayılar doğrudan "yıllık sıklık" üzerine yorumlanır.

# %% 5a. Önce yanlış yol: sıklık üzerine doğrusal regresyon
formul_s = ("VehPower + VehAge + DrivAge + BonusMalus + logDensity"
            " + C(VehGas) + C(Area)")
lm_s = smf.wls("siklik ~ " + formul_s, sig, weights=sig.Exposure).fit()
print(f"Doğrusal model: {(lm_s.fittedvalues < 0).sum():,} poliçe için NEGATİF hasar sıklığı tahmini")

# %% 5b. Poisson GLM
glm_s = smf.glm("ClaimNb ~ " + formul_s, sig, family=sm.families.Poisson(),
                offset=np.log(sig.Exposure)).fit()
ga = np.exp(glm_s.conf_int())
glm_tablo = pd.DataFrame({"ağırlık": glm_s.params, "exp(ağırlık)": np.exp(glm_s.params),
                          "GA alt": ga[0], "GA üst": ga[1]}).round(4)
print(glm_tablo)
print(f"\nBonusMalus +10 puan → sıklık {np.exp(10 * glm_s.params['BonusMalus']):.2f} katı")
print(f"Benzinli (Regular) araç → dizele göre {np.exp(glm_s.params['C(VehGas)[T.Regular]']):.2f} katı")
print(f"Nüfus yoğunluğu 2 katına çıkınca → sıklık {2 ** glm_s.params['logDensity']:.3f} katı")
# Son satıra dikkat: log(Density) kullandığımız için yorum "yoğunluk %X artınca sıklık %Y artar"
# biçimindedir (esneklik). Dönüşüm yorumu değiştirir – kitabın uyarısı.

# %% [markdown]
# ## 6. Kırma 1: Aşırı yayılım (overdispersion)
#
# Poisson'un katı varsayımı: varyans = ortalama. Gerçek veride varyans genellikle daha büyüktür
# (gözlemlenmeyen risk farkları: dikkatsiz sürücü, kötü yol...).
# Test: Pearson χ² / serbestlik derecesi ≈ 1 olmalı. 1'den büyükse aşırı yayılım var.
# Sonuç: Katsayılar hâlâ tutarlıdır, ama STANDART HATALAR KÜÇÜK HESAPLANIR →
# güven aralıkları dar, p-değerleri sahte biçimde anlamlı. Yorumun güvenilirliği çöker.
# Çözüm: Quasi-Poisson (SE'yi ölçekle) veya Negatif Binom dağılımı.

# %% 6. Aşırı yayılım testi ve düzeltme
phi = glm_s.pearson_chi2 / glm_s.df_resid
print(f"Pearson χ² / sd = {phi:.2f}   (Poisson varsayımı: 1)")

qp = smf.glm("ClaimNb ~ " + formul_s, sig, family=sm.families.Poisson(),
             offset=np.log(sig.Exposure)).fit(scale="X2")      # quasi-Poisson
nb = smf.glm("ClaimNb ~ " + formul_s, sig, family=sm.families.NegativeBinomial(alpha=1.0),
             offset=np.log(sig.Exposure)).fit()
se = pd.DataFrame({"Poisson SE": glm_s.bse, "Quasi-Poisson SE": qp.bse, "Neg. Binom SE": nb.bse,
                   "Poisson z": glm_s.tvalues, "Quasi z": qp.tvalues}).round(4)
print(se)
print(f"AIC  Poisson: {glm_s.aic:,.0f}   Negatif Binom: {nb.aic:,.0f}  (küçük olan daha iyi)")

# %% [markdown]
# ## 7. Sürücü yaşı: doğrusal mı?
#
# Aktüerler bilir: genç sürücüler çok riskli, risk 30-50 yaş arasında düşer,
# yaşlılıkta yeniden biraz artar. Doğrusal GLM bunu tek bir eğimle özetlemeye çalışır.
# Karşılaştırma (kitaptaki 4 yöntemin 3'ü):
#   - Doğrusal terim
#   - Kategorize (yaş grupları)
#   - GAM: f(yaş) spline ile öğrenilir (pygam, Poisson dağılımı, exposure ile)

# %% 7. Yaş etkisinin üç modeli
yas = np.arange(18, 91)
taban = {"VehPower": 6, "VehAge": 5, "BonusMalus": 50, "logDensity": sig.logDensity.median(),
         "VehGas": "Regular", "Area": "C"}
tahmin_df = pd.DataFrame({**taban, "DrivAge": yas})

# Doğrusal
p_lin = glm_s.predict(tahmin_df)

# Kategorize
sig["YasGrup"] = pd.cut(sig.DrivAge, [17, 21, 25, 30, 40, 50, 60, 70, 100])
glm_kat = smf.glm("ClaimNb ~ " + formul_s.replace("DrivAge", "C(YasGrup)"), sig,
                  family=sm.families.Poisson(), offset=np.log(sig.Exposure)).fit()
tahmin_df["YasGrup"] = pd.cut(tahmin_df.DrivAge, sig.YasGrup.cat.categories)
p_kat = glm_kat.predict(tahmin_df)

# GAM: hız için 200.000 poliçelik rastgele örnek; diğer öznitelikler doğrusal terim
orn = sig.sample(200_000, random_state=1)
X_gam = np.column_stack([orn.DrivAge, orn.BonusMalus, orn.VehPower, orn.VehAge, orn.logDensity,
                         (orn.VehGas == "Regular").astype(float)])
# λ (lam): düzgünlük cezası. Büyüdükçe eğri düzleşir; birkaç değer AIC ile denenip 1 seçildi.
gam_s = PoissonGAM(s(0, n_splines=12, lam=1) + l(1) + l(2) + l(3) + l(4) + l(5)).fit(
    X_gam, orn.ClaimNb.values, exposure=orn.Exposure.values)
X_izgara = np.column_stack([yas, np.full_like(yas, 50), np.full_like(yas, 6), np.full_like(yas, 5),
                            np.full(len(yas), sig.logDensity.median()), np.ones(len(yas))]).astype(float)
p_gam = gam_s.predict(X_izgara)

gozlenen = sig.groupby(pd.cut(sig.DrivAge, range(18, 92, 3)), observed=True).apply(
    lambda g: g.ClaimNb.sum() / g.Exposure.sum())
orta = [iv.mid for iv in gozlenen.index]

# Modelleri karşılaştırılabilir kılmak için her eğriyi 45 yaşındaki sürücüye oranlıyoruz:
# "Bu yaştaki sürücü, 45 yaşındaki benzer bir sürücüye göre kaç kat riskli?"
i45 = int(np.where(yas == 45)[0][0])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(orta, gozlenen.values, "o-", color="gray", ms=4)
axes[0].set(title="Ham veri: yaşa göre gözlenen sıklık", xlabel="Sürücü yaşı",
            ylabel="Hasar / poliçe-yıl")
axes[1].plot(yas, p_lin / p_lin.iloc[i45], lw=2, label="Doğrusal GLM")
axes[1].step(yas, p_kat / p_kat.iloc[i45], where="mid", lw=2, label="Kategorize GLM")
axes[1].plot(yas, p_gam / p_gam[i45], lw=2.5, label="GAM (spline)")
axes[1].axhline(1, color="gray", lw=1)
axes[1].set(title="Modeller: 45 yaşa göre risk çarpanı (diğer her şey sabit)",
            xlabel="Sürücü yaşı", ylabel="Göreli risk")
axes[1].legend(fontsize=8)
kaydet("04_surucu_yasi_gam.png")
print(f"18 yaşındaki sürücü, 45 yaşındakine göre:  doğrusal GLM {p_lin.iloc[0] / p_lin.iloc[i45]:.2f} kat"
      f"  |  GAM {p_gam[0] / p_gam[i45]:.2f} kat")
# Yorum:
# - Doğrusal GLM gençleri 45 yaşındakilerden DAHA AZ riskli gösteriyor: tek eğim her şeyi kaçırıyor.
# - GAM karmaşık bir şekil buluyor: 18 yaşında yüksek risk, 25-30 arası en düşük, 40-50 arasında
#   ikinci bir tepe. Bu tepe için aktüerya literatüründe sık öne sürülen olası açıklama:
#   poliçe ebeveynin adına, aracı ise genç çocuk kullanıyor. Model bunu bilemez; biz yorumlarız.
# - Sol paneldeki ham fark, sağdaki model farkından çok daha büyük. Neden? Genç sürücülerin
#   BonusMalus puanı da yüksek (yeni sürücüler 100'den başlar). "Diğer her şey sabitken" yaşın
#   kendi etkisi, ham farkın bir kısmı. Ham grafik ile model yorumu aynı şeyi söylemez!
print(f"AIC  doğrusal GLM: {glm_s.aic:,.0f}   kategorize GLM: {glm_kat.aic:,.0f}")

# %% [markdown]
# ## 8. Tuzak: log(y) modellemek ≠ log bağlantı fonksiyonu
#
# Çok sık karıştırılır:
#   - OLS ile log(y) modellemek  →  E[log y]'yi modeller
#   - GLM ile log bağlantı       →  log E[y]'yi modeller
# Jensen eşitsizliği: E[log y] ≤ log E[y]. Bu yüzden log(y) modelinin tahminini exp() ile geri
# çevirince ORTALAMAYI SİSTEMATİK OLARAK DÜŞÜK tahmin ederiz (retransformation bias).
# (Not: Sigorta verisinde y=0 çok olduğu için log(y) hiç alınamaz bile. Bisiklet verisiyle gösteriyoruz.)

# %% 8. Bisiklet verisinde iki yaklaşım
f_b = "C(season) + C(weather) + C(workday) + temp + hum + windspeed"
ols_log = smf.ols("np.log(cnt) ~ " + f_b, bike).fit()
glm_log = smf.glm("cnt ~ " + f_b, bike, family=sm.families.Poisson()).fit()
print(f"Gerçek toplam kiralama:                 {bike.cnt.sum():>12,.0f}")
print(f"OLS log(y) → exp(tahmin) toplamı:       {np.exp(ols_log.fittedvalues).sum():>12,.0f}")
print(f"Poisson GLM (log bağlantı) toplamı:     {glm_log.fittedvalues.sum():>12,.0f}")
print("→ Log bağlantılı GLM ortalamayı (ve toplamı) korur; log(y) modeli düşük tahmin eder.")

# %% [markdown]
# ## 9. Kırma 2: Area ve Density
#
# Area (A-F), aslında nüfus yoğunluğunun kategorize edilmiş hâlidir. İkisini birlikte koymak
# Bölüm 6'daki çoklu bağlantının GLM'deki karşılığıdır. GAM'lerde bunun doğrusal olmayan
# karşılığına "concurvity" denir. Sonuç: aynı bilgi iki öznitelik arasında paylaşılır,
# katsayılar tuhaflaşır ve yorumlanamaz hâle gelir.

# %% 9. Korelasyon ve katsayı karşılaştırması
alan_kod = sig.Area.map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6})
print(f"Spearman korelasyonu (Area kodu, Density): {alan_kod.corr(sig.Density, method='spearman'):.3f}")

yalniz_area = smf.glm("ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(VehGas) + C(Area)",
                      sig, family=sm.families.Poisson(), offset=np.log(sig.Exposure)).fit()
k = pd.DataFrame({"yalnız Area": np.exp(yalniz_area.params.filter(like="Area")),
                  "Area + logDensity": np.exp(glm_s.params.filter(like="Area"))}).round(3)
print(k)
print("→ Yalnız Area varken A'dan E'ye risk belirgin biçimde artıyor (E: %55 daha riskli).")
print("  Density eklenince Area katsayıları 1'e doğru çöküyor, F 1'in altına iniyor ve hiçbiri")
print("  anlamlı kalmıyor (güven aralıkları 1'i kapsıyor). Bilgi kaybolmadı, iki öznitelik arasında paylaşıldı.")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(2, 7), k["yalnız Area"], "o-", label="Yalnız Area")
ax.plot(range(2, 7), k["Area + logDensity"], "s-", label="Area + logDensity")
ax.axhline(1, color="gray", lw=1)
ax.set_xticks(range(2, 7), list("BCDEF"))
ax.set(title="Aynı bilgiyi taşıyan iki öznitelik: katsayılar çöker",
       xlabel="Bölge (A = referans)", ylabel="exp(ağırlık)")
ax.legend()
kaydet("05_area_density.png")

# %% [markdown]
# ## Özet (sunum için)
# 1. GLM = dağılım + doğrusal yordayıcı + bağlantı fonksiyonu. Lojistik ve doğrusal regresyon da birer GLM.
# 2. Sayım verisinde doğrusal model negatif tahmin yapar; Poisson GLM (log bağlantı) yapmaz.
#    Offset ile farklı gözlem süreleri doğru biçimde hesaba katılır. Yorum çarpımsaldır.
# 3. Aşırı yayılım: Poisson'un varyans = ortalama varsayımı bozulunca SE'ler küçülür,
#    güven aralıkları yalan söyler. Quasi-Poisson / Negatif Binom ile düzeltilir.
# 4. Etkileşim katsayısı tek başına yorumlanmaz; ana etkiyle birlikte okunur.
# 5. GAM, sürücü yaşının U şeklindeki etkisini veriden öğrenir; doğrusal model 18 yaşındaki
#    sürücünün riskini ciddi biçimde hafife alır.
# 6. log(y) ≠ log bağlantı: biri E[log y], diğeri log E[y] modeller (Jensen eşitsizliği).
# 7. Aynı bilgiyi taşıyan öznitelikler (Area–Density) GLM'de de yorumu bozar (GAM'de: concurvity).
