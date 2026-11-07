# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Açıklamaları Değerlendirmek: Bölüm 32'nin Taiwan Kredi Verisinde Uygulaması
#
# **Öğrenci:** Berkay Çongar (20269258007) · **Bölüm:** 32 — Yorumlanabilirlik Yöntemlerinin Değerlendirilmesi
#
# Bölüm 32'nin ana iddiası şu: Gerçek veride bir açıklamanın "doğru cevabı" (temel gerçeklik) yoktur.
# Bu yüzden açıklamaları, bölümün saydığı **özellikler** üzerinden dolaylı ölçütlerle değerlendiririz.
# Bu defter, Doshi-Velez ve Kim'in (2017) **işlev düzeyinde değerlendirme** yaklaşımını (insan gerektirmeyen vekil ölçütler)
# zorlu bir gerçek veri setine uygular: 30.000 kredi kartı müşterisi, 23 öznitelik, dengesiz sınıflar ve güçlü çoklu bağlantı.
#
# Değerlendirilen açıklama yöntemleri: **LIME** (yerel vekil), **KernelSHAP** (modelden bağımsız Shapley),
# **TreeSHAP** (ağaçlara özgü Shapley) ve **küresel vekil karar ağacı**.
#
# | Bölümdeki özellik | Bu defterdeki ölçüt |
# |---|---|
# | Algoritmik karmaşıklık | Açıklama başına süre |
# | Doğruluk ve sadakat | Vekil ağacın gerçek etikete (AUC) ve kara kutuya (R²) uyumu; LIME'ın yerel R²'si ve açıklanan örnekteki hatası; SHAP toplamsallığı |
# | Anlaşılabilirlik | Açıklama boyutu (öznitelik ya da yaprak sayısı) ile sadakat arasındaki ödünleşim |
# | Kararlılık | Aynı örnekte tekrarlanan çalıştırmalarda ve komşu müşterilerde ilk 5 özniteliğin korunması (Jaccard@5) |
# | Tutarlılık | Benzer başarımlı modellerin açıklamaları arasındaki sıra korelasyonu (Rashomon etkisi) |
# | Belirlilik | Model belirsizliği ile açıklama belirsizliği arasındaki ilişki (bootstrap) |
# | Önem derecesi | Açıklamadaki ağırlığın en önemli birkaç öznitelikte toplanması |
# | Yenilik | Eğitim dağılımından uzak örneklerde açıklamanın güvenilirliği |
# | Temsil edicilik | Yerel bir açıklamanın, örnekten uzaklaştıkça geçerliliğini yitirmesi |
#
# **Çalıştırma:** Bu defterdeki çıktılar ve README'deki sayılar `HIZLI = False` ile üretildi (iki çekirdekli bir makinede yaklaşık 5 dakika).
# Ders içi canlı gösterimde aşağıdaki hücrede `HIZLI = True` yapın; daha az örnekle aynı analiz yaklaşık 1 dakikada çalışır.

# %%
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

try:  # Jupyter'de tablolar HTML olarak, düz Python'da metin olarak gösterilir
    from IPython.display import display
except ImportError:
    display = print

HIZLI = False         # canlı gösterim için True (≈1 dk) · raporlanan sonuçlar: False
RS = 42               # tekrarlanabilirlik için tohum

N_ORNEK = 40 if HIZLI else 150      # açıklanacak test örneği sayısı
N_KARARLILIK = 15 if HIZLI else 40  # kararlılık deneylerindeki örnek sayısı
N_TEKRAR = 5 if HIZLI else 10       # aynı örnekte tekrar sayısı
N_KOMSU = 8 if HIZLI else 20        # girdi kararlılığı için komşu sayısı
B_BOOT = 5 if HIZLI else 15         # bootstrap model sayısı
KSHAP_NS = 500 if HIZLI else 2000   # KernelSHAP koalisyon örneği
LIME_NS = 2000 if HIZLI else 5000   # LIME pertürbasyon örneği

RENK = {"LIME": "#DD6B20", "KernelSHAP": "#2B6CB0", "TreeSHAP": "#2F855A", "Vekil ağaç": "#6B46C1"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
rng = np.random.default_rng(RS)

SEKIL_KLASORU = Path("sekiller")
SEKIL_KLASORU.mkdir(exist_ok=True)


def sekil_kaydet(ad):
    """Grafiği README ve sunum için 200 dpi PNG olarak da kaydeder."""
    plt.savefig(SEKIL_KLASORU / f"{ad}.png", dpi=200, bbox_inches="tight")

# %% [markdown]
# ## 1. Veri: Taiwan kredi kartı temerrüdü
#
# Kaynak: Yeh ve Lien (2009), UCI Machine Learning Repository, CC BY 4.0. Hedef: müşterinin bir sonraki ay temerrüde düşmesi.
# Temizlik adımları: belgelenmemiş `EDUCATION` kodları (0, 5, 6) ve `MARRIAGE` kodu (0) "diğer" sınıfına alındı;
# veri setindeki tutarsız ad `PAY_0`, ayların sırasına uysun diye `PAY_1` yapıldı.

# %%
VERI_YOLU = Path("veri/default_of_credit_card_clients.csv")
YEDEK_KAYNAKLAR = [
    # PR birleştirildikten sonra ders deposundaki kopya
    "https://raw.githubusercontent.com/mesutpolatgil/xai-dersi-2026/main/2026-guz/"
    "32_evaluation-of-interpretability-methods/uygulama/berkaycongar-20269258007/veri/default_of_credit_card_clients.csv",
    # UCI'deki özgün .xls dosyasının birebir kopyası (okumak için xlrd gerekir)
    "https://raw.githubusercontent.com/jphall663/interpretable_machine_learning_with_python/master/default_of_credit_card_clients.xls",
]


def veri_yukle():
    if VERI_YOLU.exists():
        ham = pd.read_csv(VERI_YOLU)
    else:
        ham = None
        for url in YEDEK_KAYNAKLAR:
            try:
                ham = pd.read_excel(url, header=1) if url.endswith(".xls") else pd.read_csv(url)
                break
            except Exception as hata:  # bir kaynak erişilemezse sıradakini dene
                print("Kaynak okunamadı:", url, "→", hata)
        if ham is None:
            raise FileNotFoundError("Veri bulunamadı: veri/ klasörünü ya da internet bağlantısını kontrol edin.")
    assert ham.shape == (30000, 25), "UCI veri setinin 30.000 satır ve 25 sütunu olmalı"
    df = ham.drop(columns="ID").rename(columns={"PAY_0": "PAY_1", "default payment next month": "TEMERRUT"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


df = veri_yukle()
X_df, y = df.drop(columns="TEMERRUT"), df["TEMERRUT"].to_numpy()
OZN = list(X_df.columns)
d = len(OZN)
print(f"{len(df):,} müşteri · {d} öznitelik · temerrüt oranı %{100 * y.mean():.2f}")

# %% [markdown]
# **Bu veri neden zorlu?** Altı aylık fatura tutarları (`BILL_AMT1-6`) ve ödeme durumları (`PAY_1-6`) birbirine çok bağlı.
# Böyle durumda bir model aynı bilgiyi farklı öznitelikler arasında keyfî biçimde paylaştırabilir. Açıklamaların kararlılığı ve
# tutarlılığı tam da burada sınanır.

# %%
kor = X_df.corr().to_numpy()
fig, ax = plt.subplots(figsize=(7.2, 6))
im = ax.imshow(kor, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(d), OZN, rotation=90, fontsize=8)
ax.set_yticks(range(d), OZN, fontsize=8)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson korelasyonu")
bill = [OZN.index(f"BILL_AMT{i}") for i in range(1, 7)]
pay = [OZN.index(f"PAY_{i}") for i in range(1, 7)]
ust = lambda idx: kor[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
ax.set_title(f"Öznitelik korelasyonları\nBILL_AMT blok ortalaması r = {ust(bill).mean():.2f} · "
             f"PAY blok ortalaması r = {ust(pay).mean():.2f}", fontsize=10.5)
plt.tight_layout()
sekil_kaydet("01-korelasyon")
plt.show()

# %% [markdown]
# ## 2. Modeller
#
# Kara kutu olarak gradyan artırmalı ağaçları (GBM) kullanıyoruz. Tutarlılık deneyi için üç model daha eğitiyoruz: GBM'e yakın
# başarımlı bir rastgele orman ve bir sinir ağı, karşılaştırma için de daha basit bir lojistik regresyon. Tüm modeller
# standartlaştırılmış özniteliklerle çalışır; açıklamalar da bu ölçekte hesaplanır ve olasılık birimindedir.

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X_df.to_numpy(float), y, test_size=0.3, random_state=RS, stratify=y)
olcek = StandardScaler().fit(X_tr)
Xtr, Xte = olcek.transform(X_tr), olcek.transform(X_te)

modeller = {
    "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=3, subsample=0.8, random_state=RS),
    "Rastgele orman": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=RS),
    "Sinir ağı": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=RS),
    "Lojistik regresyon": LogisticRegression(max_iter=2000),
}
satirlar = []
for ad, m in modeller.items():
    t0 = time.perf_counter()
    m.fit(Xtr, y_tr)
    p = m.predict_proba(Xte)[:, 1]
    satirlar.append({"Model": ad, "Test AUC": roc_auc_score(y_te, p), "Doğruluk": accuracy_score(y_te, p > 0.5),
                     "Eğitim (sn)": time.perf_counter() - t0})
basarim = pd.DataFrame(satirlar).set_index("Model").round(3)
display(basarim)

kara_kutu = modeller["GBM"]
f = lambda X: kara_kutu.predict_proba(np.atleast_2d(X))[:, 1]

# Açıklanacak örnekler: test kümesinden sabit tohumla seçilir
idx = rng.choice(len(Xte), N_ORNEK, replace=False)
Xa = Xte[idx]

# %% [markdown]
# ## 3. Açıklama yöntemleri ve yöntem özellikleri
#
# Bölüm 32, açıklama *yöntemleri* için dört özellik sayar. İlk üçü yöntemin tasarımından gelir; dördüncüsünü ölçüyoruz.
#
# | Yöntem | İfade gücü (çıktının dili) | Yarı saydamlık | Taşınabilirlik |
# |---|---|---|---|
# | LIME | Yerel doğrusal modelin ağırlıkları (eğim) | Yok: yalnızca girdi değiştirip tahmine bakar | Her model |
# | KernelSHAP | Shapley değerleri (katkı, toplamı = tahmin − ortalama) | Yok | Her model |
# | TreeSHAP | Shapley değerleri | Yüksek: ağaçların iç yapısını okur | Yalnızca ağaç modelleri |
# | Vekil karar ağacı | EĞER-İSE kuralları | Yok | Her model |

# %%
arka_plan = shap.kmeans(Xtr, 25)                      # KernelSHAP için özetlenmiş arka plan
arka_plan_agac = shap.sample(Xtr, 100, random_state=RS)  # TreeSHAP (girişimsel) için arka plan
lime_ac = LimeTabularExplainer(Xtr, feature_names=OZN, mode="classification",
                               discretize_continuous=False, sample_around_instance=True, random_state=RS)


def lime_acikla(x, model=None, tohum=RS, ns=LIME_NS, k=None):
    """LIME: x çevresinde örnekler üretir, ağırlıklı doğrusal model kurar.
    Döndürür: ağırlık vektörü, yerel R², sabit terim, yerel modelin x'teki tahmini."""
    model = model or kara_kutu
    lime_ac.random_state = np.random.RandomState(tohum)
    e = lime_ac.explain_instance(x, model.predict_proba, num_features=k or d, num_samples=ns, labels=(1,))
    w = np.zeros(d)
    for j, v in e.as_map()[1]:
        w[j] = v
    return w, e.score, e.intercept[1], float(np.ravel(e.local_pred)[0])


def kshap_aciklayici(model):
    return shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], arka_plan)


def kshap_acikla(aciklayici, X, tohum=RS, ns=KSHAP_NS):
    """KernelSHAP. l1_reg=False: shap'in varsayılan öznitelik seçimini kapatır, 23 özniteliğin hepsi için tahmin döner."""
    np.random.seed(tohum)  # KernelSHAP koalisyonları rastgele örnekler
    return np.atleast_2d(np.array(aciklayici.shap_values(np.atleast_2d(X), nsamples=ns, l1_reg=False, silent=True)))


ks_gbm = kshap_aciklayici(kara_kutu)
ts_gbm = shap.TreeExplainer(kara_kutu, data=arka_plan_agac, model_output="probability",
                            feature_perturbation="interventional")
tshap_acikla = lambda X: np.atleast_2d(ts_gbm.shap_values(np.atleast_2d(X)))

# Algoritmik karmaşıklık: açıklama başına süre
sure = {}
t0 = time.perf_counter(); A_lime = np.array([lime_acikla(x)[0] for x in Xa]); sure["LIME"] = (time.perf_counter() - t0) / N_ORNEK
t0 = time.perf_counter(); A_ks = kshap_acikla(ks_gbm, Xa); sure["KernelSHAP"] = (time.perf_counter() - t0) / N_ORNEK
t0 = time.perf_counter(); A_ts = tshap_acikla(Xa); sure["TreeSHAP"] = (time.perf_counter() - t0) / N_ORNEK
print("Açıklama başına süre (ms):", {k: round(1000 * v, 1) for k, v in sure.items()})

# %% [markdown]
# **Aynı müşteri, üç açıklama.** Aşağıdaki grafik bölümün çıkış noktasını gösteriyor: Üç yöntem aynı tahmini açıklıyor ama
# farklı şeyler söylüyor. Hangisinin "doğru" olduğunu söyleyecek bir temel gerçeklik yok. Bu yüzden özelliklerle değerlendirmemiz gerekiyor.

# %%
i0 = int(np.argmax(f(Xa)))  # açıklanan örnekler içinde temerrüt olasılığı en yüksek müşteri
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
for ax, (ad, A) in zip(axes, [("LIME", A_lime), ("KernelSHAP", A_ks), ("TreeSHAP", A_ts)]):
    sira = np.argsort(-np.abs(A[i0]))[:8][::-1]
    ax.barh([OZN[j] for j in sira], A[i0, sira], color=RENK[ad])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_title(ad)
    ax.set_xlabel("Atıf (olasılık birimi)")
fig.suptitle(f"Aynı müşteri (tahmin edilen temerrüt olasılığı {f(Xa[i0])[0]:.2f}) için üç açıklama", y=1.02)
plt.tight_layout()
sekil_kaydet("02-ayni-musteri-uc-aciklama")
plt.show()

# %% [markdown]
# ## 4. Doğruluk ve sadakat
#
# * **Sadakat:** Açıklama kara kutunun tahminine ne kadar iyi yaklaşıyor? Vekil ağaç için test kümesinde
#   $R^2\big(f(x), g(x)\big)$, yani ağacın kara kutu olasılıklarını ne kadar iyi taklit ettiği.
# * **Doğruluk:** Açıklama görülmemiş veriyi ne kadar iyi tahmin ediyor? Vekil ağacın **gerçek etiketlere** göre AUC'si.
#
# Ağaç derinleştikçe sadakat artar ama yaprak sayısı, yani anlaşılabilirlik bedeli de büyür.

# %%
f_tr, f_te = f(Xtr), f(Xte)
vekil = []
for derinlik in range(1, 9):
    g = DecisionTreeRegressor(max_depth=derinlik, min_samples_leaf=50, random_state=RS).fit(Xtr, f_tr)
    gp = g.predict(Xte)
    vekil.append({"Derinlik": derinlik, "Yaprak": g.get_n_leaves(), "Sadakat R²": r2_score(f_te, gp),
                  "Doğruluk AUC": roc_auc_score(y_te, gp)})
vekil = pd.DataFrame(vekil)
auc_kk = roc_auc_score(y_te, f_te)

# LIME'ın yerel sadakati: K öznitelikli açıklamanın yerel R²'si (anlaşılabilirlik ödünleşimi)
K_LISTE = [1, 2, 3, 5, 8, 12, d]
lime_k = {k: [lime_acikla(x, k=k)[1] for x in Xa[:N_KARARLILIK]] for k in K_LISTE}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(vekil["Yaprak"], vekil["Sadakat R²"], "o-", color=RENK["Vekil ağaç"], label="Sadakat (kara kutuya R²)")
a1.plot(vekil["Yaprak"], vekil["Doğruluk AUC"], "s--", color="#4A5568", label="Doğruluk (gerçek etikete AUC)")
a1.axhline(auc_kk, color="#2B6CB0", ls=":", label=f"Kara kutu AUC = {auc_kk:.3f}")
a1.set_xscale("log", base=2)
a1.set_xlabel("Vekil ağacın yaprak sayısı (log ölçek)")
a1.set_title("Küresel vekil ağaç: boyut ↑, sadakat ↑")
for _, r in vekil.iterrows():
    a1.annotate(f"d={int(r['Derinlik'])}", (r["Yaprak"], r["Sadakat R²"]), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8)
a1.legend(fontsize=8.5, loc="lower right")
ort = [np.median(lime_k[k]) for k in K_LISTE]
alt = [np.percentile(lime_k[k], 25) for k in K_LISTE]
ust_ = [np.percentile(lime_k[k], 75) for k in K_LISTE]
a2.fill_between(K_LISTE, alt, ust_, color=RENK["LIME"], alpha=0.2, label="çeyrekler arası")
a2.plot(K_LISTE, ort, "o-", color=RENK["LIME"], label="medyan yerel R²")
a2.set_xlabel("LIME açıklamasındaki öznitelik sayısı (K)")
a2.set_ylabel("Yerel sadakat (ağırlıklı R²)")
a2.set_title("LIME: kısa açıklama, düşük yerel sadakat")
a2.legend(fontsize=8.5)
plt.tight_layout()
sekil_kaydet("03-sadakat-anlasilabilirlik")
plt.show()

hata_ts = np.abs(A_ts.sum(1) + ts_gbm.expected_value - f(Xa)).max()
hata_ks = np.abs(A_ks.sum(1) + ks_gbm.expected_value - f(Xa)).max()
lime_x_hatasi = np.array([abs(lime_acikla(x)[3] - f(x)[0]) for x in Xa])
print(f"SHAP toplamsallığı (yerel doğruluk): en büyük hata TreeSHAP {hata_ts:.1e}, KernelSHAP {hata_ks:.1e}")
print(f"LIME (tüm öznitelikler) medyan yerel R² = {np.median(lime_k[d]):.2f}")
print(f"LIME'ın yerel modelinin açıklanan müşterinin kendisindeki hatası |f(x) − g(x)|: "
      f"medyan {np.median(lime_x_hatasi):.3f}, en büyük {lime_x_hatasi.max():.3f} (olasılık birimi)")
print("Derinlik 3 vekil ağacın kuralları:\n" + export_text(
    DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=RS).fit(Xtr, f_tr), feature_names=OZN, decimals=2))

# %% [markdown]
# **Yorum.** Bu veri setinde kara kutu büyük ölçüde son ayın ödeme durumuna (`PAY_1`) dayandığı için, derinliği 3 olan (8 yapraklı)
# bir ağaç bile davranışın büyük kısmını okunabilir kurallarla yakalıyor. Ağaç büyüdükçe sadakat artıyor ama kuralları okumak zorlaşıyor.
# SHAP yöntemleri tanım gereği toplamsaldır, yani atıfların toplamı tahmine eşittir. Bu, sadakatin *bir* biçimidir ama açıklamanın
# modelin "neden" öyle karar verdiğini doğru yakaladığını garanti etmez.
#
# LIME'da durum daha çarpıcı: Yerel modelin R²'si düşük. Dahası, açıklanan müşterinin **kendisi için** verdiği tahmin bile kara kutudan
# belirgin biçimde sapıyor. Nedeni boyut sayısı: LIME her öznitelikte bir standart sapmalık gürültüyle örnek üretir. 23 boyutta bu
# örnekler müşteriden ortalama $\sqrt{23} \approx 4{,}8$ standart sapma uzağa düşer. Varsayılan çekirdek genişliği
# ($0{,}75\sqrt{23} \approx 3{,}6$) bu uzak örnekleri neredeyse eşit ağırlıklandırır. Sonuçta "yerel" model, müşterinin çevresini değil
# geniş ve gerçekte olmayan bir bölgeyi özetler.
#
# ## 5. Kararlılık
#
# **(a) Tekrar kararlılığı:** Aynı örneği aynı yöntemle farklı rastgele tohumlarla açıklarız ve en önemli 5 özniteliğin
# kümeleri arasındaki ortalama Jaccard benzerliğine bakarız ($|A\cap B|/|A\cup B|$, 1 = hep aynı). Bölüm, deterministik olmayan
# bileşenleri (LIME'ın örneklemesi gibi) kararsızlığın bir kaynağı olarak sayar.
#
# **(b) Girdi kararlılığı:** Bölümün tanımıyla, bir örneğin özniteliklerindeki küçük değişiklikler açıklamayı önemli ölçüde
# değiştirmemelidir. Müşteriyi çok az değiştirerek komşular üretiriz (her öznitelikte $\epsilon \sim N(0,\,0{,}05^2)$ standart sapma)
# ve açıklamanın ilk 5 özniteliğinin ne kadar aynı kaldığını yine Jaccard@5 ile ölçeriz. LIME ve KernelSHAP'ta tohumu sabit tutarız;
# böylece fark yalnızca girdideki değişimden gelir. Karşılaştırma için modelin tahminindeki değişimi de raporlarız: Tahmin neredeyse
# hiç değişmiyorsa açıklamanın da değişmemesi beklenir.

# %%
def jaccard_top(A, k=5):
    kume = [set(np.argsort(-np.abs(a))[:k]) for a in A]
    return np.mean([len(kume[i] & kume[j]) / len(kume[i] | kume[j])
                    for i in range(len(kume)) for j in range(i + 1, len(kume))])


Xk = Xa[:N_KARARLILIK]
tekrar = {"LIME": [], "KernelSHAP": [], "TreeSHAP": []}
for x in Xk:
    tekrar["LIME"].append(jaccard_top([lime_acikla(x, tohum=t)[0] for t in range(N_TEKRAR)]))
    tekrar["KernelSHAP"].append(jaccard_top([kshap_acikla(ks_gbm, x, tohum=t)[0] for t in range(N_TEKRAR)]))
    tekrar["TreeSHAP"].append(jaccard_top([tshap_acikla(x)[0] for _ in range(2)]))

def jaccard_cift(a, b, k=5):
    A, B = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(b))[:k])
    return len(A & B) / len(A | B)


girdi = {"LIME": [], "KernelSHAP": [], "TreeSHAP": []}
tahmin_degisimi = []
for x in Xk:
    komsular = x + rng.normal(0, 0.05, size=(N_KOMSU, d))
    e0 = {"LIME": lime_acikla(x)[0], "KernelSHAP": kshap_acikla(ks_gbm, x)[0], "TreeSHAP": tshap_acikla(x)[0]}
    E = {"LIME": np.array([lime_acikla(z)[0] for z in komsular]),
         "KernelSHAP": kshap_acikla(ks_gbm, komsular), "TreeSHAP": tshap_acikla(komsular)}
    for ad in girdi:
        girdi[ad].append(np.mean([jaccard_cift(e0[ad], e) for e in E[ad]]))
    tahmin_degisimi.append(np.mean(np.abs(f(komsular) - f(x))))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
b = a1.boxplot([tekrar[k] for k in tekrar], patch_artist=True, widths=0.5)
a1.set_xticks(range(1, len(tekrar) + 1), list(tekrar))
for kutu, ad in zip(b["boxes"], tekrar):
    kutu.set_facecolor(RENK[ad]); kutu.set_alpha(0.6)
a1.set_ylim(0, 1.05)
a1.set_ylabel("Jaccard@5 (1 = her çalıştırmada aynı)")
a1.set_title(f"Tekrar kararlılığı ({N_TEKRAR} farklı tohum)")
ad3 = list(girdi)
b = a2.boxplot([girdi[k] for k in ad3], patch_artist=True, widths=0.5)
a2.set_xticks(range(1, 4), ad3)
for kutu, ad in zip(b["boxes"], ad3):
    kutu.set_facecolor(RENK[ad]); kutu.set_alpha(0.6)
a2.set_ylim(0, 1.05)
a2.set_ylabel("Jaccard@5 (1 = komşularda da aynı)")
a2.set_title(f"Girdi kararlılığı (tahmin ort. {np.mean(tahmin_degisimi):.3f} değişirken)")
plt.tight_layout()
sekil_kaydet("04-kararlilik")
plt.show()
print("Medyan Jaccard@5, tekrar:", {k: round(float(np.median(v)), 2) for k, v in tekrar.items()})
print("Medyan Jaccard@5, girdi :", {k: round(float(np.median(v)), 2) for k, v in girdi.items()})
print(f"Komşularda tahmin edilen olasılığın ortalama mutlak değişimi: {np.mean(tahmin_degisimi):.4f}")

# %% [markdown]
# **Yorum.** TreeSHAP deterministiktir; aynı girdiye her zaman aynı açıklamayı verir. LIME'ın ve sınırlı sayıda koalisyon
# örnekleyen KernelSHAP'ın sonucu ise tohuma bağlıdır. Girdi kararlılığında modelin tahmini neredeyse hiç değişmezken
# açıklamanın ilk 5'i değişiyorsa, kararsızlık modelden değil açıklama yönteminden gelir. İki kararlılık türü farklı şeyler ölçer:
# Bir yöntem tohuma karşı kararlı, girdiye karşı kararsız olabilir; bunun tersi de mümkündür.
#
# ## 6. Tutarlılık: Rashomon etkisi
#
# Aynı görevde eğitilmiş, benzer tahminler üreten modellerin açıklamaları ne kadar benzer? Tüm modelleri **aynı yöntemle**
# (KernelSHAP, aynı arka plan) açıklarız. Böylece fark yöntemden değil modellerden gelir. Solda modellerin tahminleri arasındaki
# korelasyon, sağda açıklamaları arasındaki ortalama Spearman sıra korelasyonu yer alıyor.

# %%
adlar = list(modeller)
A_model = {ad: (A_ks if ad == "GBM" else kshap_acikla(kshap_aciklayici(modeller[ad]), Xa)) for ad in adlar}
P = {ad: modeller[ad].predict_proba(Xte)[:, 1] for ad in adlar}
M = len(adlar)
tahmin_kor, aciklama_kor = np.eye(M), np.eye(M)
for i in range(M):
    for j in range(i + 1, M):
        tahmin_kor[i, j] = tahmin_kor[j, i] = np.corrcoef(P[adlar[i]], P[adlar[j]])[0, 1]
        rho = [spearmanr(a, b)[0] for a, b in zip(A_model[adlar[i]], A_model[adlar[j]])]
        aciklama_kor[i, j] = aciklama_kor[j, i] = np.nanmean(rho)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.3]})
for ax, Mx, baslik in [(axes[0], tahmin_kor, "Tahminler (Pearson r)"), (axes[1], aciklama_kor, "Açıklamalar (ort. Spearman ρ)")]:
    ax.imshow(Mx, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(M), [a.replace(" ", "\n") for a in adlar], fontsize=8.5)
    ax.set_yticks(range(M), adlar, fontsize=8.5)
    ax.grid(False)
    for i in range(M):
        for j in range(M):
            ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", color="white" if Mx[i, j] > 0.6 else "#1A202C", fontsize=9)
    ax.set_title(baslik)
onem = pd.DataFrame({ad: np.abs(A_model[ad]).mean(0) for ad in adlar}, index=OZN)
onem = onem / onem.sum()
ilk = onem.mean(1).sort_values(ascending=False).index[:8]
onem.loc[ilk[::-1]].plot.barh(ax=axes[2], width=0.8, color=["#2B6CB0", "#2F855A", "#C05621", "#718096"])
axes[2].set_xlabel("Ortalama |SHAP| payı")
axes[2].set_title("Her model hangi özniteliklere dayanıyor?")
axes[2].legend(fontsize=8)
plt.tight_layout()
sekil_kaydet("05-tutarlilik-rashomon")
plt.show()
ust_ucgen = np.triu_indices(M, 1)
print(f"Tahminler arası korelasyon: {tahmin_kor[ust_ucgen].min():.2f}–{tahmin_kor[ust_ucgen].max():.2f} · "
      f"açıklamalar arası sıra korelasyonu: {aciklama_kor[ust_ucgen].min():.2f}–{aciklama_kor[ust_ucgen].max():.2f}")

# %% [markdown]
# **Yorum.** Tahminler birbirine yakın olsa da açıklamalar daha az örtüşür. Bölümün uyardığı gibi yüksek tutarlılık ancak modeller
# gerçekten aynı ilişkilere dayanıyorsa beklenir. Çoklu bağlantılı `PAY` ve `BILL_AMT` blokları, aynı bilginin modeller arasında farklı
# özniteliklere dağıtılmasına izin verir; sağdaki grafikte rastgele ormanın `PAY_2`'ye, lojistik regresyonun `BILL_AMT1`'e daha çok
# dayandığı görülüyor. KernelSHAP'ın tekrar kararlılığı yüksek olduğundan (bölüm 5), farkın büyük kısmı yöntemin gürültüsünden değil
# modellerin kendisinden geliyor. Yani buradaki düşük tutarlılık açıklamanın hatası değil, Rashomon etkisinin bir göstergesi.
#
# ## 7. Belirlilik
#
# Açıklama, modelin ne kadar emin olduğunu yansıtıyor mu? Modelin belirsizliğini, eğitim verisinden bootstrap ile üretilmiş
# $B$ GBM'in tahminlerinin standart sapmasıyla ölçüyoruz. Her modelde TreeSHAP açıklamalarını da hesaplıyoruz; açıklamanın
# belirsizliği, atıfların modeller arasındaki standart sapmasının ortalamasıdır.

# %%
boot_modeller = []
for b in range(B_BOOT):
    s = rng.choice(len(Xtr), len(Xtr), replace=True)
    boot_modeller.append(GradientBoostingClassifier(n_estimators=100, max_depth=3, subsample=0.8, random_state=b).fit(Xtr[s], y_tr[s]))
boot_tahmin = np.array([m.predict_proba(Xa)[:, 1] for m in boot_modeller])
boot_aciklama = np.array([shap.TreeExplainer(m, data=arka_plan_agac, model_output="probability",
                                             feature_perturbation="interventional").shap_values(Xa) for m in boot_modeller])
model_belirsizligi = boot_tahmin.std(0)
aciklama_belirsizligi = boot_aciklama.std(0).mean(1)
rho_b = spearmanr(model_belirsizligi, aciklama_belirsizligi)[0]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
sc_ = ax.scatter(model_belirsizligi, aciklama_belirsizligi, c=boot_tahmin.mean(0), cmap="viridis", s=30)
fig.colorbar(sc_, ax=ax, label="Ortalama temerrüt olasılığı")
ax.set_xlabel("Model belirsizliği (tahminlerin std'si)")
ax.set_ylabel("Açıklama belirsizliği (atıfların ort. std'si)")
ax.set_title(f"Belirsiz tahminin açıklaması da belirsiz (Spearman ρ = {rho_b:.2f})")
plt.tight_layout()
sekil_kaydet("06-belirlilik")
plt.show()

# %% [markdown]
# **Yorum.** Tek bir SHAP grafiği bu belirsizliği göstermez; aynı müşteri için tek bir çubuk dizisi verir. Bölümün önerdiği gibi açıklama,
# modelin ve kendisinin belirsizliğini de taşımalıdır; örneğin bootstrap aralıklarıyla.
#
# ## 8. Önem derecesi
#
# Açıklama, hangi öznitelik ya da koşulun en önemli olduğunu açıkça gösteriyor mu? Atıf yöntemleri öznitelikleri sıralar. Bir
# açıklamanın ağırlığının ne kadarının en önemli 3 öznitelikte toplandığını ölçüyoruz. Vekil ağacın bir yaprağa giden kural yolu
# ise koşulları sıralamaz; bu da önem derecesinin açıklamanın *biçimine* bağlı olduğunu gösterir.

# %%
pay3 = lambda A: np.sort(np.abs(A), 1)[:, ::-1][:, :3].sum(1) / (np.abs(A).sum(1) + 1e-12)
onem_derecesi = {"LIME": pay3(A_lime), "KernelSHAP": pay3(A_ks), "TreeSHAP": pay3(A_ts)}
print("En önemli 3 özniteliğin toplam |atıf| içindeki medyan payı:",
      {k: f"%{100 * np.median(v):.0f}" for k, v in onem_derecesi.items()})

# %% [markdown]
# ## 9. Yenilik
#
# Açıklanan örnek eğitim dağılımından uzaksa, model ve açıklama güvenilmez olabilir. Yeniliği bir Isolation Forest skoruyla ölçüyoruz.
# Test kümesindeki en "yeni" örnekleri tipik örneklerle karşılaştırıyoruz.

# %%
iso = IsolationForest(n_estimators=300, random_state=RS).fit(Xtr)
yenilik = -iso.score_samples(Xte)
n_y = N_KARARLILIK
yeni_idx = np.argsort(-yenilik)[:n_y]
orta = np.argsort(yenilik)[len(yenilik) // 4: 3 * len(yenilik) // 4]
tipik_idx = rng.choice(orta, n_y, replace=False)


def grup_olc(I):
    X_ = Xte[I]
    bt = np.array([m.predict_proba(X_)[:, 1] for m in boot_modeller]).std(0)
    r2 = [lime_acikla(x)[1] for x in X_]
    jac = [jaccard_top([lime_acikla(x, tohum=t)[0] for t in range(3)]) for x in X_]
    return bt, np.array(r2), np.array(jac)


tip, yen = grup_olc(tipik_idx), grup_olc(yeni_idx)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, k, baslik in zip(axes, range(3), ["Model belirsizliği (bootstrap std)", "LIME yerel R²", "LIME tekrar kararlılığı (Jaccard@5)"]):
    b = ax.boxplot([tip[k], yen[k]], patch_artist=True, widths=0.5)
    ax.set_xticks([1, 2], ["Tipik", "Yeni (dağılım dışı)"])
    for kutu, renk in zip(b["boxes"], ["#A0AEC0", "#E53E3E"]):
        kutu.set_facecolor(renk); kutu.set_alpha(0.6)
    ax.set_title(baslik, fontsize=10)
plt.suptitle("Yenilik: eğitim verisinden uzak müşterilerde açıklama ne kadar güvenilir?", y=1.03)
plt.tight_layout()
sekil_kaydet("07-yenilik")
plt.show()
display(pd.DataFrame({"Tipik (medyan)": [np.median(t) for t in tip], "Yeni (medyan)": [np.median(t) for t in yen]},
                     index=["Model belirsizliği", "LIME yerel R²", "LIME Jaccard@5"]).round(3))

# %% [markdown]
# **Yorum.** Bölümün dediği gibi yenilik ve belirlilik birbirine bağlıdır: Verinin seyrek olduğu bölgede model daha belirsizdir.
# LIME'ın kendi kalite göstergesi olan yerel R² ise yeni müşterilerde düşmüyor. Yani açıklama, örneğin "yeni" olduğunu kendiliğinden
# söylemiyor; tipik bir müşteri için olduğu gibi aynı özgüvenle bir açıklama üretiyor. Sadakat ölçütü iyi görünse bile yeniliği ayrıca
# ölçmek gerekir.
#
# ## 10. Temsil edicilik
#
# Bir açıklama kaç örneği kapsar? Bir müşteri için kurulan yerel doğrusal açıklamayı, o müşteriden giderek uzaklaşan test örneklerinde
# deneriz. Hata büyüdükçe açıklamanın temsil ettiği bölge daralır. Dört açıklamayı karşılaştırıyoruz:
#
# * **Gerçek komşu modeli (başvuru çizgisi):** Müşteriye en yakın %1'lik *gerçek* eğitim örneklerine kurulmuş doğrusal model.
#   Yerel bir açıklamanın nasıl davranması gerektiğini gösterir: yakında iyi, uzakta kötü.
# * **LIME:** Örneklerini veri dağılımına bakmadan, müşterinin çevresinde geniş bir bölgeden üretir.
# * **Küresel vekil ağaç (derinlik 4)** ve **sabit tahmin** (eğitimdeki ortalama olasılık).

# %%
vekil4 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=50, random_state=RS).fit(Xtr, f_tr)
dilimler = [(0, 1), (1, 5), (5, 20), (20, 50), (50, 100)]
sonuc = {"Gerçek komşu modeli (yerel)": [], "LIME (yerel)": [], "Vekil ağaç (küresel)": [], "Sabit tahmin": []}
for x0 in Xa[: min(10, N_ORNEK)]:
    w, _, b0, _ = lime_acikla(x0)
    z = (Xte - lime_ac.scaler.mean_) / lime_ac.scaler.scale_       # LIME doğrusal modelini kendi ölçeğinde kurar
    g_lime = b0 + z @ w
    komsu = np.argsort(np.linalg.norm(Xtr - x0, axis=1))[: len(Xtr) // 100]
    g_komsu = Ridge(alpha=1.0).fit(Xtr[komsu], f_tr[komsu]).predict(Xte)
    sira = np.argsort(np.linalg.norm(Xte - x0, axis=1))
    satir = {k: [] for k in sonuc}
    for lo, hi in dilimler:
        I = sira[int(lo / 100 * len(sira)): int(hi / 100 * len(sira))]
        satir["Gerçek komşu modeli (yerel)"].append(np.mean(np.abs(f_te[I] - g_komsu[I])))
        satir["LIME (yerel)"].append(np.mean(np.abs(f_te[I] - g_lime[I])))
        satir["Vekil ağaç (küresel)"].append(np.mean(np.abs(f_te[I] - vekil4.predict(Xte[I]))))
        satir["Sabit tahmin"].append(np.mean(np.abs(f_te[I] - f_tr.mean())))
    for k in sonuc:
        sonuc[k].append(satir[k])

etiket = [f"%{lo}–{hi}" for lo, hi in dilimler]
fig, ax = plt.subplots(figsize=(8, 4.2))
for k, renk, stil in zip(sonuc, ["#1A202C", RENK["LIME"], RENK["Vekil ağaç"], "#A0AEC0"], ["o-", "o-", "o-", "o--"]):
    ax.plot(etiket, np.median(sonuc[k], 0), stil, color=renk, label=k)
ax.set_xlabel("Açıklanan müşteriye en yakın test örnekleri (uzaklık yüzdelik dilimi)")
ax.set_ylabel("Ortalama mutlak hata |f(x) − g(x)|")
ax.set_title("Yerel açıklama yakında iyi, uzakta kötü; LIME ise yakında bile kötü")
ax.legend(fontsize=8.5)
plt.tight_layout()
sekil_kaydet("08-temsil-edicilik")
plt.show()
display(pd.DataFrame({k: np.median(v, 0) for k, v in sonuc.items()}, index=etiket).round(3))

# %% [markdown]
# **Yorum.** Gerçek komşularla kurulan yerel model, bölümdeki "temsil edicilik" kavramını açıkça gösteriyor: Açıklanan müşterinin
# yakınında kara kutuya en iyi yaklaşan odur, uzaklaştıkça hatası büyür. Bu, tek bir müşteriyi temsil eden bir açıklamanın beklenen
# davranışıdır. Küresel vekil ağaç her yerde aynı orta düzey hatayı verir, yani bütün veriyi temsil eder. LIME'ın yerel modeli ise
# müşterinin en yakın komşularında bile sabit bir tahminden daha kötü. Örnekleri gerçek veri dağılımından değil, geniş ve yapay bir
# bölgeden geldiği için LIME'ın açıklaması ne yereli ne de bütünü temsil ediyor. Bu, dersin sızma testi bakışına uyan bir zafiyettir:
# Varsayılan ayarlarla kullanılan LIME, yüksek boyutlu tablo verisinde "yerel" adını hak etmeyebilir.
#
# ## 11. Karne: yöntemleri özelliklere göre yan yana koymak
#
# Tek bir "en iyi" yöntem yoktur. Her yöntem bazı özelliklerde güçlü, bazılarında zayıftır. Bölüm 32'nin değerlendirme
# çerçevesinin pratik çıktısı, kullanım amacına göre hangi özelliklerin önemli olduğuna karar verip yöntemleri bu özelliklere göre seçmektir.

# %%
vekil_d3 = vekil.set_index("Derinlik").loc[3]
karne = pd.DataFrame({
    "Süre (ms/açıklama)": [1000 * sure["LIME"], 1000 * sure["KernelSHAP"], 1000 * sure["TreeSHAP"], 0.0],
    "Sadakat": [f"yerel R² {np.median(lime_k[d]):.2f}; x'te hata {np.median(lime_x_hatasi):.2f}", f"toplamsal (hata {hata_ks:.0e})",
                f"toplamsal (hata {hata_ts:.0e})", f"küresel R² {vekil_d3['Sadakat R²']:.2f}"],
    "Tekrar kararlılığı (Jaccard@5)": [np.median(tekrar["LIME"]), np.median(tekrar["KernelSHAP"]), np.median(tekrar["TreeSHAP"]), 1.0],
    "Girdi kararlılığı (Jaccard@5)": [np.median(girdi["LIME"]), np.median(girdi["KernelSHAP"]),
                                       np.median(girdi["TreeSHAP"]), np.nan],
    "İlk 3'ün payı": [np.median(onem_derecesi["LIME"]), np.median(onem_derecesi["KernelSHAP"]),
                      np.median(onem_derecesi["TreeSHAP"]), np.nan],
    "Boyut": [f"{d} ağırlık", f"{d} katkı", f"{d} katkı", f"{int(vekil_d3['Yaprak'])} yaprak"],
    "Temsil": ["tek örnek (yerel)", "tek örnek", "tek örnek", "tüm veri (küresel)"],
    "Yarı saydamlık": ["yok", "yok", "yüksek", "yok"],
    "Taşınabilirlik": ["her model", "her model", "ağaçlar", "her model"],
}, index=["LIME", "KernelSHAP", "TreeSHAP", "Vekil ağaç (d=3)"]).round(2)
display(karne)

# %% [markdown]
# ## 12. Değerlendirme düzeyleri açısından bu defter
#
# Buradaki tüm ölçütler **işlev düzeyindedir**: Hiçbiri insan gerektirmez, hepsi ucuz ve tekrarlanabilir. Ama hiçbiri bir kredi
# uzmanının ya da başvurunun reddedildiği müşterinin açıklamayı *anlayıp* doğru karar verip vermediğini ölçmez. Bu soruyu yanıtlamak
# için **insan düzeyinde** (örneğin "bu açıklamaya bakarak modelin kararını tahmin edin" deneyi) ya da **uygulama düzeyinde**
# (gerçek kredi sürecine yerleştirip test etme) değerlendirme gerekir. Bölüm 32'nin ana mesajı da budur: Vekil ölçütler gereklidir ama
# yeterli değildir.
