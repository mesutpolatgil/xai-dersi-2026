# %% [markdown]
# # Bölüm 7 – Lojistik Regresyon: Uygulama
#
# Ahmet Batuhan Kaya – Açıklanabilir Yapay Zeka (XAI), 2026 Güz
#
# Akış:
#   0. Kurulum
#   1. Köprü: Kitaptaki penguen örneğinin Python'da yeniden üretimi
#   2. Neden doğrusal regresyon değil? (Kitaptaki tümör örneği)
#   3. Zorlu veri: German Credit (1.000 kredi başvurusu, %30 temerrüt)
#   4. Yorumlama: log-odds, odds oranı, güven aralığı, tekil başvuru
#   5. Tuzak 1: Odds oranı ≠ risk oranı
#   6. Kırma 1: Tam ayrışma (complete separation) – katsayı sonsuza gider
#   7. Kırma 2: Non-collapsibility – ilgisiz değişken eklemek katsayıyı değiştirir
#   8. Olasılıklar güvenilir mi? Kalibrasyon
#
# VS Code'da her "# %%" satırının üstünde "Run Cell" yazar; hücre hücre çalıştırın.

# %% 0. Kurulum
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)
SEKIL_KLASORU = "sekiller"
os.makedirs(SEKIL_KLASORU, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(os.path.join(SEKIL_KLASORU, ad), dpi=150, bbox_inches="tight")
    plt.show()


def etiket(ad):
    """C(x, Treatment('a'))[T.b]  ->  x: b"""
    if ad.startswith("C("):
        return ad[2:].split(",")[0] + ": " + ad.split("[T.")[1].rstrip("]")
    return ad


# %% [markdown]
# ## 1. Köprü: Kitaptaki penguen modeli
#
# Kitap, Chinstrap penguenlerinde vücut ölçülerinden cinsiyeti tahmin ediyor.
# Kilo (body_mass_g) üç gruba bölünmüş: Smol / Regular / Absolute_Unit.
# Python'da lojistik regresyon: statsmodels'in smf.logit() fonksiyonu.

# %% 1. Penguen verisi
peng = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
peng = peng[peng["species"] == "Chinstrap"].dropna().copy()
q25, q75 = peng["body_mass_g"].quantile([0.25, 0.75])
peng["chonkiness"] = np.select([peng["body_mass_g"] <= q25, peng["body_mass_g"] <= q75],
                               ["Smol_Penguin", "Regular_Penguin"], "Absolute_Unit")
peng["disi"] = (peng["sex"].str.upper() == "FEMALE").astype(int)    # Kitaptaki gibi P(dişi)

peng_model = smf.logit("disi ~ bill_length_mm + bill_depth_mm + flipper_length_mm"
                       " + C(chonkiness, Treatment('Smol_Penguin'))", data=peng).fit(disp=0)
peng_tablo = pd.DataFrame({"ağırlık": peng_model.params, "SE": peng_model.bse,
                           "odds oranı": np.exp(peng_model.params)}).drop("Intercept").round(2)
# Sabit terim çok büyük (kitapta da öyle): gaga ve yüzgeç uzunluğu 0 olan, gerçekçi olmayan bir penguen.
peng_tablo.index = [etiket(i) for i in peng_tablo.index]
print(peng_tablo)
print(f"\nGaga uzunluğu odds oranı = {peng_tablo.loc['bill_length_mm', 'odds oranı']}"
      " < 1  → gaga uzadıkça dişi olma odds'u AZALIR.")
print("Kitap bunu 'increases by a factor of ...' diye yazmış: kitaptaki yorum hatası!")

# %% [markdown]
# ## 2. Neden doğrusal regresyon değil?
#
# Sınıfları 0 ve 1 diye kodlayıp doğrusal regresyon kurarsak:
#   - Tahminler 0'ın altına ve 1'in üstüne çıkar → olasılık olamaz.
#   - Uç noktalar eklemek çizgiyi kaydırır → 0,5 eşiği bozulur.
# Lojistik regresyon, doğrusal kombinasyonu sigmoid fonksiyonundan geçirir:
#   P(Y=1) = 1 / (1 + exp(−(β0 + β1·x)))     → her zaman 0 ile 1 arasında.

# %% 2. Kitaptaki tümör örneği: doğrusal vs lojistik
x1 = np.array([1, 2, 3, 8, 9, 10, 11, 9]); y1 = np.array([0, 0, 0, 1, 1, 1, 1, 0])
x2 = np.r_[x1, 7, 7, 7, 20, 19, 5, 5, 4, 4.5]; y2 = np.r_[y1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
izgara = np.linspace(0, 21, 200)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (x, y, ad) in zip(axes, [(x1, y1, "Temel veri"), (x2, y2, "Uç noktalar eklendi")]):
    b1, b0 = np.polyfit(x, y, 1)
    lr = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y)
    ax.scatter(x, y + RNG.normal(0, 0.02, len(y)), color="black", s=20, zorder=3)
    ax.plot(izgara, b0 + b1 * izgara, label="Doğrusal regresyon")
    ax.plot(izgara, lr.predict_proba(izgara.reshape(-1, 1))[:, 1], label="Lojistik regresyon")
    ax.axhline(0.5, ls=":", color="gray")
    ax.set(title=ad, xlabel="Tümör büyüklüğü", ylim=(-0.3, 1.4))
axes[0].set_ylabel("Kötü huylu olma (0/1)")
axes[0].legend(loc="upper left", fontsize=8)
kaydet("01_dogrusal_vs_lojistik.png")

# Yorum: Sağ panelde doğrusal çizgi yatıklaşıp 0,5 eşiğini geçtiği yer sola kayıyor ve
# 1'in üstüne taşıyor. Lojistik eğri ise iki durumda da 0-1 arasında kalıyor.

# %% [markdown]
# ## 3. Zorlu veri: German Credit
#
# Hofmann (1994), UCI Machine Learning Repository. 1.000 kredi başvurusu,
# hedef: krediyi geri ödememe (temerrüt). Neden zorlu?
#   - Dengesiz sınıflar: %70 iyi, %30 kötü.
#   - Çoğu öznitelik kategorik ve çok seviyeli (hesap bakiyesi, kredi geçmişi...).
#   - Bazı kategorilerde çok az gözlem var → geniş güven aralıkları.
#   - Gerçek bir karar: banka bu modelle kredi verip vermemeyi açıklayabilmeli.

# %% 3. Veriyi yükle
kredi = pd.read_csv("https://raw.githubusercontent.com/stedy/"
                    "Machine-Learning-with-R-datasets/master/credit.csv")
kredi["temerrut"] = (kredi["default"] == 2).astype(int)
kredi["tutar_bin"] = kredi["amount"] / 1000                 # 1.000 DM birimi: yorum kolaylaşsın
kredi = kredi.rename(columns={"months_loan_duration": "vade_ay", "age": "yas",
                              "checking_balance": "hesap", "credit_history": "gecmis",
                              "savings_balance": "birikim", "purpose": "amac"})
print(kredi[["temerrut", "vade_ay", "tutar_bin", "yas"]].describe().round(2))
print("\nTemerrüt oranı:", kredi["temerrut"].mean())

# %% 3b. Model
# Referans kategoriler bilinçli seçildi: hesap 'unknown' yerine '< 0 DM' (eksi bakiye) olsun ki
# diğer kategoriler "eksi bakiyeye göre" okunsun.
formul = ("temerrut ~ vade_ay + tutar_bin + yas"
          " + C(hesap, Treatment('< 0 DM'))"
          " + C(gecmis, Treatment('repaid'))"
          " + C(birikim, Treatment('< 100 DM'))")
model = smf.logit(formul, data=kredi).fit(disp=0)
print(model.summary2().tables[1].round(3))

# %% [markdown]
# ## 4. Yorumlama
#
# Lojistik regresyonun doğrusal olduğu ölçek log-odds'tur:
#   log( p / (1−p) ) = β0 + β1·x1 + ...
# Odds = p/(1−p):  p=0,5 → odds 1 ;  p=0,2 → odds 0,25 ;  p=0,8 → odds 4.
# x_j bir birim artınca odds exp(β_j) KATINA çıkar. Bu sayıya odds oranı denir.
#   OR > 1 → risk artar,  OR < 1 → risk azalır,  OR = 1 → etkisiz.

# %% 4a. Odds oranı tablosu ve grafiği
ga = np.exp(model.conf_int())
or_tablo = pd.DataFrame({"ağırlık (log-odds)": model.params,
                         "odds oranı": np.exp(model.params),
                         "GA alt": ga[0], "GA üst": ga[1]}).drop("Intercept")
or_tablo.index = [etiket(i) for i in or_tablo.index]
print(or_tablo.round(3))

sira = or_tablo["odds oranı"].sort_values().index
fig, ax = plt.subplots(figsize=(8, 6.5))
o = or_tablo.loc[sira]
ax.errorbar(o["odds oranı"], range(len(o)),
            xerr=[o["odds oranı"] - o["GA alt"], o["GA üst"] - o["odds oranı"]], fmt="o", capsize=3)
ax.axvline(1, color="red", lw=1)
ax.set_xscale("log")
ax.set_yticks(range(len(o)), sira)
ax.set(title="Temerrüt için odds oranları (%95 GA, log ölçek)", xlabel="Odds oranı (1 = etkisiz)")
kaydet("02_odds_oranlari.png")

print(f"\nÖrnek: Diğer her şey sabitken vade 1 ay uzayınca temerrüt odds'u "
      f"{or_tablo.loc['vade_ay', 'odds oranı']:.3f} katına çıkar.")
print(f"12 ay uzayınca: {or_tablo.loc['vade_ay', 'odds oranı'] ** 12:.2f} katı "
      "(çarpımsal: 12 kez çarpılır, 12 ile çarpılmaz!)")
print(f"Hesap bakiyesi bilinmiyor (unknown) olanlarda odds, eksi bakiyelilere göre "
      f"{or_tablo.loc['hesap: unknown', 'odds oranı']:.2f} katı.")

# Dikkat, sezgiye aykırı iki sonuç:
#   - 'gecmis: critical' (kritik geçmiş) riski AZALTIYOR (OR < 1),
#   - 'gecmis: fully repaid' (tamamen ödemiş) riski ARTIRIYOR (OR > 1).
# İki olası açıklama, ikisi de XAI için ders niteliğinde:
#   1) Seçilim yanlılığı: banka kritik geçmişlilerden yalnızca en güvenilirlerine kredi vermiş olabilir.
#   2) Veri hatası: Grömping (2019), yaygın kullanılan bu veri setinde bazı kategori etiketlerinin
#      yanlış kodlandığını gösterdi ve düzeltilmiş "South German Credit" verisini yayımladı.
# Yorumlanabilir model sayesinde bu tuhaflığı GÖRDÜK. Bir kara kutu bunu sessizce öğrenirdi.
# Model veriyi açıklar, dünyayı değil.

# %% 4b. Aynı odds oranı, farklı olasılık değişimi
# Odds oranı sabittir ama olasılığa etkisi başlangıç riskine bağlıdır (sigmoid eğriliği).
or_ = 2.0
for p0 in [0.05, 0.30, 0.50, 0.90]:
    odds1 = p0 / (1 - p0) * or_
    p1 = odds1 / (1 + odds1)
    print(f"Başlangıç olasılığı {p0:.2f} → OR=2 sonrası {p1:.3f}  (artış: {100*(p1-p0):+.1f} puan)")

# %% 4c. Tek bir başvurunun açıklanması (log-odds ölçeğinde katkılar)
# Doğrusal modeldeki "fiş" mantığı burada log-odds için geçerli.
i = int(model.predict().argmax())          # Modele göre en riskli başvuru
X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
katki = (X.loc[i] * model.params)
katki.index = [etiket(k) for k in katki.index]
katki = katki[katki != 0].sort_values()
print(f"Başvuru #{i}: tahmin edilen temerrüt olasılığı = {model.predict()[i]:.3f}, "
      f"gerçek = {kredi.loc[i, 'temerrut']}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(katki.index, katki.values, color=["tab:green" if v < 0 else "tab:red" for v in katki])
ax.set(title=f"Başvuru #{i}: log-odds katkıları (kırmızı = riski artırır)", xlabel="β_j · x_j")
kaydet("03_tekil_basvuru.png")

# %% [markdown]
# ## 5. Tuzak 1: Odds oranı risk oranı değildir
#
# "Odds oranı 2" ≠ "risk 2 katı". İkisi yalnızca olay NADİR iken (p küçük) yakındır.
# German Credit'te temerrüt %30: nadir değil. Gazetelerde ve hatta makalelerde
# sık yapılan hata budur.

# %% 5. Hesap: eksi bakiye vs bakiyesi bilinmeyen grup
grup = kredi.groupby("hesap")["temerrut"].mean()
p_a, p_b = grup["< 0 DM"], grup["unknown"]
odds_orani = (p_a / (1 - p_a)) / (p_b / (1 - p_b))
risk_orani = p_a / p_b
print(f"Temerrüt oranı: eksi bakiye %{100*p_a:.1f}, bilinmiyor %{100*p_b:.1f}")
print(f"Risk oranı (RR) = {risk_orani:.2f}   |   Odds oranı (OR) = {odds_orani:.2f}")
print("Doğru cümle: 'Eksi bakiyelilerde temerrüt riski {:.1f} kat' — {:.1f} kat değil.".format(risk_orani, odds_orani))

# %% [markdown]
# ## 6. Kırma 1: Tam ayrışma (complete separation)
#
# Bir öznitelik sınıfları KUSURSUZ ayırırsa, olasılığı en yüksek yapan eğim sonsuzdur:
# sigmoid bir basamak fonksiyonuna dönüşmek ister. Optimizasyon yakınsamaz,
# katsayı ve standart hata patlar. Kitabın önerisi: ceza (L2/ridge) veya önsel dağılım.
#
# Gerçekçi senaryo: Bankada "kara listede mi?" diye bir değişken var ve
# kara listedeki herkes temerrüde düşmüş. Bunu ekleyelim.

# %% 6. Tam ayrışma deneyi
kredi_ay = kredi.copy()
# Temerrüde düşenlerin rastgele %20'si kara listede; temerrüde düşmeyen hiç kimse değil.
kredi_ay["kara_liste"] = ((kredi_ay["temerrut"] == 1) & (RNG.random(len(kredi_ay)) < 0.2)).astype(int)
print(pd.crosstab(kredi_ay["kara_liste"], kredi_ay["temerrut"]))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m_ayr = smf.logit(formul + " + kara_liste", data=kredi_ay).fit(disp=0, maxiter=200)
print(f"\nstatsmodels: β(kara_liste) = {m_ayr.params['kara_liste']:.1f}, "
      f"SE = {m_ayr.bse['kara_liste']:.1f}, odds oranı = {np.exp(m_ayr.params['kara_liste']):.2e}")
print("→ Anlamsız derecede büyük katsayı, devasa standart hata. Model 'çok önemli' diyor ama")
print("  güven aralığı her şeyi kapsıyor: yorum çöktü.")

# Çözüm: Ceza ekle. C küçüldükçe ceza büyür. Ceza azaldıkça katsayının nasıl sınırsız büyüdüğünü göster.
Xs = pd.get_dummies(kredi_ay[["vade_ay", "tutar_bin", "yas", "hesap", "gecmis", "birikim", "kara_liste"]],
                    drop_first=True, dtype=float)
Xs = (Xs - Xs.mean()) / Xs.std()
Cs = np.logspace(-2, 4, 25)
yol = []
for c_deger in Cs:   # dikkat: "C" adını kullanmıyoruz, formüldeki C() ile çakışır
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(C=c_deger, max_iter=5000).fit(Xs, kredi_ay["temerrut"])
    yol.append(lr.coef_[0][list(Xs.columns).index("kara_liste")])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(Cs, yol, marker="o", ms=3)
ax.set_xscale("log")
ax.set(title="Tam ayrışma: ceza azaldıkça katsayı durmadan büyür",
       xlabel="C  (sağa doğru ceza azalır)", ylabel="β kara_liste (standartlaştırılmış)")
kaydet("04_tam_ayrisma.png")

# %% [markdown]
# ## 7. Kırma 2: Non-collapsibility (Mood, 2010)
#
# Doğrusal regresyonda, x ile İLİŞKİSİZ bir değişken eklemek x'in katsayısını değiştirmez.
# Lojistik regresyonda DEĞİŞTİRİR. Çünkü katsayılar gizli bir hata varyansına göre ölçeklenir;
# yeni değişken açıklanmamış varyansı azaltınca bütün katsayılar büyür.
# Sonuç: Farklı değişken setleriyle kurulmuş iki lojistik modelin katsayıları
# (veya iki farklı çalışmanın odds oranları) doğrudan karşılaştırılamaz.
#
# Simülasyonla kanıtlayalım (x ve z bağımsız, gerçek β_x = 1):

# %% 7. Simülasyon
n = 20000
x = RNG.normal(size=n)
z = RNG.normal(size=n)                                 # x ile korelasyonu ~0
p = 1 / (1 + np.exp(-(1.0 * x + 2.0 * z)))
yb = RNG.binomial(1, p)
yc = 1.0 * x + 2.0 * z + RNG.logistic(size=n)          # aynı gizli değişkenin sürekli hâli
sim = pd.DataFrame({"x": x, "z": z, "yb": yb, "yc": yc})
print(f"corr(x, z) = {np.corrcoef(x, z)[0, 1]:.3f}")

lin_x = smf.ols("yc ~ x", sim).fit().params["x"]
lin_xz = smf.ols("yc ~ x + z", sim).fit().params["x"]
log_x = smf.logit("yb ~ x", sim).fit(disp=0).params["x"]
log_xz = smf.logit("yb ~ x + z", sim).fit(disp=0).params["x"]

sonuc = pd.DataFrame({"yalnız x": [lin_x, log_x], "x + z": [lin_xz, log_xz]},
                     index=["Doğrusal β_x", "Lojistik β_x"]).round(3)
print(sonuc)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([0, 1], [lin_x, lin_xz], width=0.35, label="Doğrusal")
ax.bar([0.4, 1.4], [log_x, log_xz], width=0.35, label="Lojistik")
ax.axhline(1, color="red", ls="--", lw=1, label="Gerçek β_x = 1")
ax.set_xticks([0.2, 1.2], ["Model: yalnız x", "Model: x + z"])
ax.set(title="İlişkisiz z eklemek: doğrusal sabit, lojistik değişir", ylabel="β_x kestirimi")
ax.legend(fontsize=8)
kaydet("05_non_collapsibility.png")

# %% [markdown]
# ## 8. Olasılıklar güvenilir mi? Kalibrasyon
#
# Kitap: "Olasılıkların kalibre olup olmadığını kontrol etmelisiniz: %60 gerçekten %60 mı?"
# Test verisinde tahminleri 10 kutuya böl, her kutuda ortalama tahmin ile gerçek oranı karşılaştır.
# Köşegene yakınsa model kalibre.

# %% 8. Eğitim/test ayrımı ve kalibrasyon
test_idx = RNG.choice(len(kredi), size=300, replace=False)
egitim = kredi.drop(index=test_idx)
test = kredi.loc[test_idx]
m_tr = smf.logit(formul, data=egitim).fit(disp=0)
p_test = m_tr.predict(test)
print(f"Test AUC = {roc_auc_score(test['temerrut'], p_test):.3f}")

gercek, tahmin = calibration_curve(test["temerrut"], p_test, n_bins=8, strategy="quantile")
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Kusursuz kalibrasyon")
ax.plot(tahmin, gercek, marker="o", label="Lojistik regresyon")
ax.set(title="Kalibrasyon (test verisi)", xlabel="Ortalama tahmin edilen olasılık",
       ylabel="Gözlenen temerrüt oranı")
ax.legend(fontsize=8)
kaydet("06_kalibrasyon.png")

# %% [markdown]
# ## Özet (sunum için)
# 1. Lojistik regresyon, doğrusal modeli sigmoid ile 0-1 arasına sıkıştırır; log-odds ölçeğinde doğrusaldır.
# 2. Katsayı → exp(β) = odds oranı. Etki çarpımsaldır; olasılıktaki değişim başlangıç riskine bağlıdır.
# 3. Odds oranı risk oranı değildir; olay sık olduğunda fark büyür (German Credit'te açıkça görülür).
# 4. Tam ayrışmada katsayı sonsuza gider; çözüm ceza/önsel dağılım.
# 5. Non-collapsibility: ilişkisiz bir değişken bile katsayıları değiştirir → modeller arası
#    odds oranı karşılaştırması tehlikelidir (Mood, 2010).
# 6. Yorumlanabilir model ≠ doğru model: 'kritik geçmiş' gibi sezgiye aykırı katsayılar
#    veri toplama sürecindeki seçilim yanlılığını yansıtabilir.
