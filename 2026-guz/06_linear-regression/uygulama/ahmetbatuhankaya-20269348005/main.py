# %% [markdown]
# # Bölüm 6 – Doğrusal Regresyon: Uygulama
#
# Ahmet Batuhan Kaya – Açıklanabilir Yapay Zeka (XAI), 2026 Güz
#
# Akış:
#   0. Kurulum
#   1. Köprü: Kitaptaki bisiklet modelinin Python'da yeniden üretimi
#   2. Zorlu veri: Ames Housing (2.930 ev, 80+ öznitelik)
#   3. Varsayım kontrolü: Ham fiyat neden sorunlu, log dönüşümü neden gerekli?
#   4. Yorumlama: Katsayılar, ağırlık grafiği, etki grafiği, tekil tahmin
#   5. Köprü: Doğrusal modelde SHAP değerleri kapalı formülle hesaplanır
#   6. Kırma: Çoklu bağlantı katsayıları nasıl bozar?
#   7. Çözüm ve yeni sorun: Lasso ve seçim kararsızlığı
#
# VS Code'da her "# %%" satırının üstünde "Run Cell" yazar; hücre hücre çalıştırın.

# %% 0. Kurulum
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso, lasso_path
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)          # Tüm rastgelelik bu tohuma bağlı: sonuçlar tekrarlanabilir
SEKIL_KLASORU = "sekiller"
os.makedirs(SEKIL_KLASORU, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def etiket(ad):
    """statsmodels'in uzun kategori adlarını okunur hâle getirir:
    C(mutfak_kalite, Treatment('TA'))[T.Ex]  ->  mutfak_kalite: Ex"""
    if ad.startswith("C("):
        return ad[2:].split(",")[0] + ": " + ad.split("[T.")[1].rstrip("]")
    return ad


def kaydet(ad):
    """Grafiği hem ekrana basar hem de sunumda kullanmak için PNG olarak kaydeder."""
    plt.tight_layout()
    plt.savefig(os.path.join(SEKIL_KLASORU, ad), dpi=150, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 1. Köprü: Kitaptaki modelin Python karşılığı
#
# Kitap R'daki lm() fonksiyonunu kullanıyor. Python'da aynı işi statsmodels'in
# formül arayüzü yapar. Neden scikit-learn değil? Çünkü sklearn standart hata,
# t-istatistiği ve güven aralığı vermez; yorumlanabilirlik için bunlar şart.
#
# Kategorik değişkenlerde referans kategoriyi kitapla aynı seçiyoruz
# (Treatment kodlama): mevsim için WINTER, hava için GOOD.

# %% 1. Bisiklet verisi ve kitaptaki model
BIKE_URL = ("https://raw.githubusercontent.com/christophM/"
            "interpretable-ml-book/master/data/bike.csv")
bike = pd.read_csv(BIKE_URL)

bike_formul = ("cnt ~ C(season, Treatment('WINTER')) + C(holiday, Treatment('N'))"
               " + C(workday, Treatment('N')) + C(weather, Treatment('GOOD'))"
               " + temp + hum + windspeed + cnt_2d_bfr")

# Kitap veriyi 2/3 eğitim - 1/3 test olarak rastgele bölüyor
egitim_idx = RNG.choice(len(bike), size=int(len(bike) * 2 / 3), replace=False)
bike_egitim = bike.iloc[egitim_idx]
bike_model = smf.ols(bike_formul, data=bike_egitim).fit()
print(bike_model.summary().tables[1])

# Kitaptaki yorumun aynısı: diğer her şey sabitken 1 °C artış kaç bisiklet ekler?
print(f"\nSıcaklık katsayısı: {bike_model.params['temp']:.1f} bisiklet / °C")
print("Not: Rastgele bölme farklı olduğu için sayılar kitaptakinden biraz farklıdır;"
      " işaretler ve büyüklük sırası aynıdır.")

# %% [markdown]
# ## 2. Zorlu veri seti: Ames Housing
#
# De Cock (2011) tarafından derlenen, ABD'nin Iowa eyaletindeki Ames şehrinde
# 2006–2010 arasında satılan 2.930 evin verisi. Neden "zorlu"?
#   - Hedef (satış fiyatı) sağa çarpık: birkaç çok pahalı ev var.
#   - Öznitelikler birbirine çok bağlı: büyük ev = çok oda = büyük garaj...
#   - Kategorik değişkenler sıralı (kalite: Po < Fa < TA < Gd < Ex).
#
# Doğrusal modelin her zayıflığını bu veride görebiliriz.

# %% 2. Ames verisini yükle ve hazırla
AMES_URL = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"
ames_ham = pd.read_csv(AMES_URL)

# Formüllerde boşluk sorun çıkardığı için sütun adlarını sadeleştiriyoruz
ames = pd.DataFrame({
    "fiyat":          ames_ham["SalePrice"],
    "yasam_alani":    ames_ham["Gr Liv Area"] * 0.0929,     # ft² -> m²
    "oda_sayisi":     ames_ham["TotRms AbvGrd"],
    "garaj_arac":     ames_ham["Garage Cars"],
    "garaj_alani":    ames_ham["Garage Area"] * 0.0929,
    "bodrum_alani":   ames_ham["Total Bsmt SF"] * 0.0929,
    "kat1_alani":     ames_ham["1st Flr SF"] * 0.0929,
    "insa_yili":      ames_ham["Year Built"],
    "genel_kalite":   ames_ham["Overall Qual"],             # 1-10 arası puan
    "arsa_alani":     ames_ham["Lot Area"] * 0.0929,
    "merkezi_klima":  ames_ham["Central Air"],              # Y / N
    "mutfak_kalite":  ames_ham["Kitchen Qual"],             # Po, Fa, TA, Gd, Ex
}).dropna()

# Kitapta da önerildiği gibi: aşırı uç değerler doğrusal modeli çok etkiler.
# Veri setinin yazarı, 4.000 ft²'den büyük 5 evi (kısmi satışlar) çıkarmayı önerir.
ames = ames[ames["yasam_alani"] < 4000 * 0.0929].reset_index(drop=True)
print(ames.shape)
print(ames.describe().T[["mean", "std", "min", "max"]].round(1))

# %% 2b. Öznitelikler arasındaki korelasyonlar: sorunun ilk işareti
sayisal = ["yasam_alani", "oda_sayisi", "garaj_arac", "garaj_alani",
           "bodrum_alani", "kat1_alani", "insa_yili", "genel_kalite", "arsa_alani"]
kor = ames[sayisal + ["fiyat"]].corr()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(kor, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(kor)), kor.columns, rotation=45, ha="right")
ax.set_yticks(range(len(kor)), kor.columns)
for i in range(len(kor)):
    for j in range(len(kor)):
        ax.text(j, i, f"{kor.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Ames: öznitelikler arası korelasyon")
kaydet("01_korelasyon.png")

# Dikkat: garaj_arac ~ garaj_alani (0.89), yasam_alani ~ oda_sayisi (0.81),
#         bodrum_alani ~ kat1_alani (0.80). Bunları 6. adımda kullanacağız.

# %% [markdown]
# ## 3. Varsayım kontrolü
#
# Kitap altı varsayım sayıyor: doğrusallık, normallik, eş varyanslılık,
# bağımsızlık, sabit öznitelikler, çoklu bağlantı yokluğu.
# Burada ikisini görsel olarak test ediyoruz:
#   - Eş varyanslılık: Artıkların (hata) yayılımı tahminle birlikte büyüyor mu?
#   - Normallik: Artıklar normal dağılıyor mu? (Q-Q grafiği)
# Bu varsayımlar bozulursa katsayılar yine hesaplanır, ama GÜVEN ARALIKLARI
# ve p-değerleri geçersiz olur. Yorumlanabilirlik iddiamızın dayanağı çöker.

# %% 3. Ham fiyatla model ve log(fiyat) ile model
ozellikler = ("yasam_alani + oda_sayisi + garaj_arac + garaj_alani + bodrum_alani"
              " + kat1_alani + insa_yili + genel_kalite + arsa_alani"
              " + C(merkezi_klima, Treatment('N'))"
              " + C(mutfak_kalite, Treatment('TA'))")

model_ham = smf.ols("fiyat ~ " + ozellikler, data=ames).fit()
model_log = smf.ols("np.log(fiyat) ~ " + ozellikler, data=ames).fit()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for satir, (model, ad) in enumerate([(model_ham, "Ham fiyat"), (model_log, "log(fiyat)")]):
    axes[satir, 0].scatter(model.fittedvalues, model.resid, s=4, alpha=0.4)
    axes[satir, 0].axhline(0, color="red", lw=1)
    axes[satir, 0].set(title=f"{ad}: artıklar vs tahmin", xlabel="Tahmin", ylabel="Artık")
    sm.qqplot(model.resid, line="s", ax=axes[satir, 1], markersize=2)
    axes[satir, 1].set(title=f"{ad}: Q-Q grafiği", xlabel="Teorik çeyreklikler",
                       ylabel="Artıkların çeyreklikleri")
kaydet("02_varsayim_kontrolu.png")

# Görsel kontrolü sayıyla destekleyelim: Breusch-Pagan testi
# H0: artıkların varyansı sabittir (eş varyanslılık). Test istatistiği ne kadar büyükse ihlal o kadar güçlü.
from statsmodels.stats.diagnostic import het_breuschpagan
for model, ad in [(model_ham, "Ham fiyat "), (model_log, "log(fiyat)")]:
    bp = het_breuschpagan(model.resid, model.model.exog)[0]
    print(f"{ad}: Breusch-Pagan istatistiği = {bp:8.1f}")
print(f"Ham fiyat modelinin negatif fiyat tahmin ettiği ev sayısı: {(model_ham.fittedvalues < 0).sum()}")

print(f"Ham fiyat  R² = {model_ham.rsquared:.3f}, düzeltilmiş R² = {model_ham.rsquared_adj:.3f}")
print(f"log(fiyat) R² = {model_log.rsquared:.3f}, düzeltilmiş R² = {model_log.rsquared_adj:.3f}")

# Yorum: Ham fiyatta artıklar "huni" şeklinde açılıyor (değişen varyans):
# pahalı evlerde hata çok daha büyük. Kitaptaki ev örneğinin birebir aynısı.
# Model ayrıca negatif fiyat tahmin edebiliyor (Bölüm 8'deki GLM motivasyonu!).
# log dönüşümü huniyi büyük ölçüde kapatıyor; Q-Q grafiğindeki sol kuyruk
# (beklenenden çok ucuza satılmış birkaç ev) hâlâ duruyor; bunu dürüstçe belirtmek gerekir.
# Bundan sonra log modelle devam ediyoruz.

# %% [markdown]
# ## 4. Yorumlama
#
# log(fiyat) modelinde katsayının yorumu değişir:
#   x_j bir birim artınca log(fiyat) β_j artar
#   => fiyat exp(β_j) katına çıkar  => yaklaşık %100·(exp(β_j) − 1) değişim.
# Bu çarpımsal yorum, Bölüm 7'deki odds oranı mantığının aynısıdır (köprü!).

# %% 4a. Katsayı tablosu (kitaptaki Tablo 6.1'in karşılığı)
tablo = pd.DataFrame({
    "ağırlık": model_log.params,
    "SE": model_log.bse,
    "|t|": model_log.tvalues.abs(),
    "% etki": 100 * (np.exp(model_log.params) - 1),
}).round(4)
tablo.loc["Intercept", "% etki"] = np.nan   # Sabit terim bir "artış" değil, temel düzeydir
print(tablo)

print(f"\nÖrnek yorum: Diğer her şey sabitken genel kalite puanındaki 1 puanlık artış, "
      f"fiyatı yaklaşık %{tablo.loc['genel_kalite', '% etki']:.1f} artırır.")
print(f"Ama oda sayısının etkisi: %{tablo.loc['oda_sayisi', '% etki']:.2f} "
      f"→ yaşam alanı sabitken oda eklemek evi küçük odalara bölmek demek!")

# %% 4b. Ağırlık grafiği: önce standartlaştırma
# Kitaptaki uyarı: Farklı ölçeklerdeki ağırlıklar karşılaştırılamaz (m² vs puan vs yıl).
# Çözüm: Sayısal öznitelikleri standartlaştırıp (ortalama 0, std 1) modeli yeniden kurmak.
ames_std = ames.copy()
ames_std[sayisal] = StandardScaler().fit_transform(ames[sayisal])
model_std = smf.ols("np.log(fiyat) ~ " + ozellikler, data=ames_std).fit()

katsayi = model_std.params.drop("Intercept")
ga = model_std.conf_int().drop("Intercept")
sira = katsayi.sort_values().index

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(katsayi[sira], range(len(sira)),
            xerr=[katsayi[sira] - ga.loc[sira, 0], ga.loc[sira, 1] - katsayi[sira]],
            fmt="o", capsize=3)
ax.axvline(0, color="gray", lw=1)
ax.set_yticks(range(len(sira)), [etiket(s) for s in sira])
ax.set(title="Ağırlık grafiği (standartlaştırılmış, %95 GA)", xlabel="log(fiyat) üzerindeki ağırlık")
kaydet("03_agirlik_grafigi.png")

# Dikkat: "mutfak_kalite: Po" çok geniş bir güven aralığına sahip. Nedeni:
print("Mutfak kalitesi dağılımı:\n", ames["mutfak_kalite"].value_counts())
# Bu kategoride yalnızca 1 ev var. Tek bir gözlemden kestirilen katsayı yorumlanamaz;
# güven aralığı bize tam olarak bunu söylüyor. Kategorileri birleştirmek bir çözüm olabilir.

# %% 4c. Etki grafiği: ağırlık × öznitelik değeri
# Kitaptaki formül: etki_j(i) = β_j · x_j(i). Kutu grafiği, etkilerin veri boyunca dağılımını gösterir.
etkiler = pd.DataFrame({c: model_log.params[c] * ames[c] for c in sayisal})

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([etkiler[c] for c in sayisal], vert=False, tick_labels=sayisal, flierprops={"markersize": 2})
ax.set(title="Etki grafiği: β_j · x_j (log ölçeğinde)", xlabel="Etki")
kaydet("04_etki_grafigi.png")

# Ağırlık grafiğiyle farkı sunumda vurgula: insa_yili'nin ağırlığı küçük görünür
# (yıl başına), ama değerleri ~2000 olduğu için etkisi devasa ve dağınıktır.

# %% 4d. Tekil tahmin açıklaması
i = int(ames["fiyat"].idxmax())          # Verideki en pahalı ev
tahmin = np.exp(model_log.fittedvalues[i])
print(f"Ev #{i}: gerçek fiyat ${ames.loc[i, 'fiyat']:,.0f}, tahmin ${tahmin:,.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([etkiler[c] for c in sayisal], vert=False, tick_labels=sayisal, flierprops={"markersize": 2})
ax.scatter(etkiler.loc[i, sayisal], range(1, len(sayisal) + 1), marker="x", color="red", s=80, zorder=3)
ax.set(title=f"Ev #{i} için etkiler (kırmızı ×)", xlabel="Etki")
kaydet("05_tekil_tahmin.png")

# %% [markdown]
# ## 5. Köprü: Doğrusal modelde SHAP
#
# Bağımsız öznitelikler varsayımıyla, doğrusal bir modelin SHAP değeri
# kapalı formülle hesaplanır (Lundberg & Lee, 2017):
#     φ_j(i) = β_j · (x_j(i) − E[x_j])
# Yani SHAP, kitaptaki "etki"nin ortalamaya göre merkezlenmiş hâlidir.
# Kitap da bunu sezer: "Öznitelikler merkezlenirse referans örnek ortalama örnek olur."
# Kontrol: φ'lerin toplamı = bu evin tahmini − ortalama tahmin olmalı.

# %% 5. SHAP formülünün doğrulanması
X_tasarim = pd.DataFrame(model_log.model.exog, columns=model_log.model.exog_names)
phi = (X_tasarim - X_tasarim.mean()) * model_log.params
phi = phi.drop(columns="Intercept")

sol = phi.loc[i].sum()
sag = model_log.fittedvalues[i] - model_log.fittedvalues.mean()
print(f"Σ φ_j = {sol:.6f}   |   f(x) − E[f(x)] = {sag:.6f}   → eşit mi? {np.isclose(sol, sag)}")
print("\nEv #{} için en büyük 5 SHAP katkısı (log ölçeği):".format(i))
print(phi.loc[i].sort_values(key=abs, ascending=False).head(5).round(3))

# %% [markdown]
# ## 6. Kırma: Çoklu bağlantı
#
# Kitabın sınırlılıklar kısmındaki iddia: "Oda sayısı, ev büyüklüğüyle birlikte
# modele girerse negatif ağırlık alabilir; korelasyon güçlüyse denklem kararsızlaşır."
# Bunu iki yolla test ediyoruz:
#   a) VIF (Varyans Şişirme Faktörü): Bir özniteliğin diğerleriyle ne kadar açıklandığı.
#      VIF_j = 1 / (1 − R²_j). Kural: > 5 dikkat, > 10 ciddi sorun.
#   b) Bootstrap: Veriyi 500 kez yeniden örnekleyip katsayıların nasıl oynadığına bakmak.

# %% 6a. VIF
X_vif = sm.add_constant(ames[sayisal])
vif = pd.Series([variance_inflation_factor(X_vif.values, k) for k in range(1, X_vif.shape[1])],
                index=sayisal, name="VIF").sort_values(ascending=False)
print(vif.round(2))

# %% 6b. Bootstrap ile katsayı kararsızlığı
# Deney: Garaj için birbirinin neredeyse kopyası iki öznitelik var (araç kapasitesi ve alan).
# Yalnızca küçük bir alt örnekle (n=150) çalışarak gerçek bir araştırmacının
# sınırlı verisini taklit ediyoruz; kararsızlık burada çıplak gözle görülür.
B, n_alt = 500, 150
boot = []
for _ in range(B):
    idx = RNG.choice(len(ames), size=n_alt, replace=True)
    m = smf.ols("np.log(fiyat) ~ " + ozellikler, data=ames.iloc[idx]).fit()
    boot.append(m.params[["garaj_arac", "garaj_alani", "oda_sayisi", "yasam_alani"]])
boot = pd.DataFrame(boot)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(boot["garaj_arac"], boot["garaj_alani"], s=6, alpha=0.5)
axes[0].axhline(0, color="gray"); axes[0].axvline(0, color="gray")
axes[0].set(title="Bootstrap katsayıları: garaj", xlabel="β garaj_arac", ylabel="β garaj_alani")
axes[1].scatter(boot["yasam_alani"], boot["oda_sayisi"], s=6, alpha=0.5, color="tab:orange")
axes[1].axhline(0, color="gray"); axes[1].axvline(0, color="gray")
axes[1].set(title="Bootstrap katsayıları: alan ve oda", xlabel="β yasam_alani", ylabel="β oda_sayisi")
kaydet("06_bootstrap_kararsizlik.png")

for c in boot.columns:
    print(f"{c:12s}: negatif çıkma oranı = %{100 * (boot[c] < 0).mean():5.1f}")

# Yorum: Noktalar negatif eğimli bir bulut oluşturur. Model garajın toplam etkisini
# bilir, ama bunu iki öznitelik arasında nasıl paylaştıracağını bilemez: biri artarken
# diğeri azalır, işaretler değişir. Tahmin hâlâ iyidir; YORUM çökmüştür.

# %% [markdown]
# ## 7. Lasso: Çözüm mü, yeni sorun mu?
#
# Lasso, kayıp fonksiyonuna λ·Σ|β_j| cezası ekler. L1 cezası köşeli olduğu için
# bazı katsayıları tam olarak sıfıra iter: öznitelik seçimi + daha yorumlanabilir model.
# λ büyüdükçe daha az öznitelik kalır. (Kitaptaki Şekil: düzenlileştirme yolu.)

# %% 7a. Lasso yolu
Z = StandardScaler().fit_transform(ames[sayisal])
y = np.log(ames["fiyat"].values)
alfalar, katsayilar, _ = lasso_path(Z, y - y.mean(), n_alphas=300)

fig, ax = plt.subplots(figsize=(8, 5))
for k, c in enumerate(sayisal):
    ax.plot(np.log(alfalar), katsayilar[k], label=c)
ax.invert_xaxis()
ax.set(title="Lasso düzenlileştirme yolu", xlabel="log(λ)  (sağa doğru ceza azalır)",
       ylabel="Standartlaştırılmış ağırlık")
ax.legend(fontsize=7, loc="upper left")
kaydet("07_lasso_yolu.png")

# Lasso'ya ilk giren öznitelikler, kitaptaki gibi, en güçlü yordayıcılardır.
giris = {c: np.log(alfalar[np.argmax(katsayilar[k] != 0)]) for k, c in enumerate(sayisal)}
print("Lasso'ya giriş sırası:", sorted(giris, key=giris.get, reverse=True))

# %% 7b. Lasso'nun da zayıflığı: seçim kararsızlığı
# Kitaptaki gibi modelde 5 öznitelik kalacak λ'yı seçiyoruz (tüm veride).
# Sonra aynı λ ile 200 bootstrap örneğinde hangi özniteliklerin seçildiğini sayıyoruz.
# Kararlı bir seçim olsaydı, seçilen 5 öznitelik her seferinde ~%100, diğerleri ~%0 olurdu.
sifirdan_farkli = (katsayilar != 0).sum(axis=0)
lam = np.median(alfalar[sifirdan_farkli == 5])   # tam 5 öznitelik bırakan λ aralığının ortası
k_lam = np.argmin(np.abs(alfalar - lam))
tam_veri_secim = [c for c, b in zip(sayisal, katsayilar[:, k_lam]) if b != 0]
print("Tüm veride seçilen 5 öznitelik:", tam_veri_secim)

secim = np.zeros(len(sayisal))
yalniz_biri = 0                                   # garaj çiftinden yalnızca biri seçildi mi?
for _ in range(200):
    idx = RNG.choice(len(ames), size=n_alt, replace=True)
    b = Lasso(alpha=lam).fit(Z[idx], y[idx]).coef_ != 0
    secim += b
    yalniz_biri += b[sayisal.index("garaj_arac")] != b[sayisal.index("garaj_alani")]
secim_orani = pd.Series(100 * secim / 200, index=sayisal).sort_values()
print(f"Garaj çiftinden yalnızca birinin seçildiği örnek oranı: %{100 * yalniz_biri / 200:.0f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(secim_orani.index, secim_orani.values)
ax.set(title=f"Lasso seçim sıklığı (200 bootstrap, λ={lam:.4f})", xlabel="Seçilme oranı (%)")
kaydet("08_lasso_secim_kararsizligi.png")
print(secim_orani.round(1))

# Yorum: Tüm veride seçilen bir öznitelik, biraz farklı bir örneklemde dışarıda kalabiliyor.
# Garaj çiftinde Lasso çoğu zaman ikisinden birini "rastgele" seçiyor.
# Çözüm önerileri: Elastic Net (L1 + L2 cezası, korelasyonlu grupları birlikte tutar)
# ve Stability Selection (Meinshausen & Bühlmann, 2010).

# %% [markdown]
# ## Özet (sunum için)
# 1. Doğrusal model kitaptaki gibi şeffaf: katsayı = "diğer her şey sabitken" etki.
# 2. Ames'te ham fiyat, eş varyanslılık varsayımını bozuyor; log dönüşümü gerekli
#    ve yorumu çarpımsal hâle getiriyor (Bölüm 7'deki odds oranına köprü).
# 3. Doğrusal modelde SHAP = merkezlenmiş etki; formülü sayısal olarak doğruladık.
# 4. Çoklu bağlantı altında katsayılar işaret değiştiriyor: tahmin sağlam, yorum çöküyor.
# 5. Lasso seyreklik getiriyor ama korelasyonlu öznitelikler arasında seçimi kararsız.
#    → "Lasso bir değişkeni seçti" demek, "o değişken önemli" demek değildir.
