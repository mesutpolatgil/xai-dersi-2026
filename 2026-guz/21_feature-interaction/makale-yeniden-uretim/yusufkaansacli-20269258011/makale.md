# Bölüm 21: Özellik Etkileşimi — Makale Yeniden Üretimi ve Eleştirel Kıyaslama Raporu

**Hazırlayan:** Yusuf Kaan Saçlı (`20269258011`)  
**Ders:** Açıklanabilir Yapay Zeka (XAI) — Güz 2026  
**İlgili Kitap Bölümü:** Christoph Molnar, *Interpretable Machine Learning*, Bölüm 21: Özellik Etkileşimi (*Feature Interaction*)

---

## 1. Makale Künyesi (Doğrulanmış)
- **Başlık:** *REPID: Regional Effect Plots with implicit Interaction Detection*
- **Yazarlar:** Julia Herbinger, Bernd Bischl, Giuseppe Casalicchio (LMU Munich & Munich Center for Machine Learning, Almanya)
- **Yıl ve Yayın Yeri:** *Proceedings of The 25th International Conference on Artificial Intelligence and Statistics (AISTATS 2022)*, PMLR 151:10209–10233, 2022.
- **Doğrulanmış Bağlantılar:**
  - **PMLR Resmi Yayın:** https://proceedings.mlr.press/v151/herbinger22a.html
  - **arXiv Ön Baskı:** https://arxiv.org/abs/2202.07254
  - **Resmi Kaynak Kodu (GitHub — R Dili):** https://github.com/JuliaHerbinger/REPID

---

## 2. Seçilen Tip ve Neden
- **Seçilen Tip:** **Seçenek 2 — Kıyaslama (*Benchmarking*)** (Bölümdeki temel algoritmanın zayıf kaldığı durumları teorik ve deneysel olarak başka algoritmalarla kıyaslayan makale).
- **Seçim Gerekçesi:** Molnar'ın 21. Bölümünde özellik etkileşimlerini ölçmek için temel standart olarak **Friedman'ın H-İstatistiği** (Friedman & Popescu, 2008) sunulur. Herbinger, Bischl ve Casalicchio (2022), H-İstatistiğinin literatürde göz ardı edilen çok kritik bir yapısal zafiyetini (**Teorem 4**) kanıtlamakta ve Bölüm 4.1'de bunu **SHAP Etkileşim İndeksi** ($I_{j,l}$), **Greenwell Etkileşim İndeksi** ve kendi önerdikleri **REPID** çerçevesiyle kıyaslamaktadır: H-İstatistiğinin paydası özelliklerin tekil ana etki büyüklüklerine bağlı olduğu için, **en yüksek H-İstatistiğine sahip özellik çifti gerçekte en güçlü etkileşime sahip çift olmayabilir**. Buna karşılık SHAP etkileşim indeksi ve normalize edilmemiş $H^*_{jk}$ ana etki büyüklüğünden etkilenmemekte; ancak girdi korelasyonu ($\rho > 0$) yükseldiğinde hem H-İstatistiği hem de SHAP etkileşim indeksi sahte etkileşim (*spurious interaction*) üretebilmektedir.

---

## 3. Makalenin 1 Sayfalık Sade Özeti
Karmaşık makine öğrenmesi modellerinde bir özelliğin tahmine yaptığı etki başka bir özelliğin aldığı değere bağlı olduğunda **özellik etkileşimi** (*feature interaction*) ortaya çıkar. Tek boyutlu Kısmi Bağımlılık Grafikleri (1B PDP), tüm veri üzerinden ortalama aldığı için güçlü etkileşimlerin varlığında zıt yönlü yerel etkileri birbirine götürür ve yanıltıcı bir küresel ortalama çizgisi üretir.

Modeldeki etkileşimleri tespit etmek için en yaygın kullanılan küresel ölçüt **Friedman'ın H-İstatistiği**'dir ($H^2_{jk}$). Bu istatistik, iki özelliğin 2 boyutlu kısmi bağımlılığından ($\text{PD}_{jk}$) tekil kısmi bağımlılıkların ($\text{PD}_j + \text{PD}_k$) çıkarılmasıyla kalan etkileşim varyansını (pay), 2 boyutlu kısmi bağımlılığın toplam varyansına (payda) böler. Herbinger vd. (2022), **Teorem 4** ile şu çarpıcı gerçeği ortaya koyar: Paydadaki $\text{Var}(\text{PD}_{jk})$ terimi, yalnızca etkileşim varyansını değil, aynı zamanda özelliklerin **tekil ana etkilerinin varyansını da** ($\text{Var}(g_j) + \text{Var}(g_k)$) içerir. Bunun iki ağır sonucu vardır:
1. **Ana Etki Yanılsaması:** İki farklı özellik çiftinin gerçek etkileşim şiddeti bire bir aynı olsa bile, tekil ana etkisi küçük olan çiftin paydası daha küçük olacağı için $H^2_{jk}$ skoru yapay olarak çok daha yüksek çıkar. Daha da kötüsü, **zayıf ana etkili bir çiftteki orta düzey bir etkileşim**, **güçlü ana etkili bir çiftteki 2 kat daha güçlü bir etkileşimden daha yüksek $H^2_{jk}$ skoru alarak sıralamayı tamamen tersine çevirebilir!**
2. **SHAP Etkileşim İndeksi ve Normalize Edilmemiş $H^*_{jk}$:** Yazarlar, Shapley etkileşim değerlerinin ($\Phi_{j,l}(\mathbf{x})$) tanımı gereği ana etkileri dışladığını; dolayısıyla mutlak SHAP etkileşimlerinin ortalaması olarak tanımladıkları küresel **SHAP Etkileşim İndeksi**'nin ($I_{j,l}$) ve payda bölmesini kaldıran **normalize edilmemiş $H^*_{jk}$** ölçütünün ana etki büyüklüğünden etkilenmediğini gösterir.
3. **Korelasyon Sınırı ve REPID:** Ancak hem H-İstatistiği hem de SHAP etkileşim indeksi özelliklerin ortak dağılımına dayandığı için, özellikler arasında yüksek korelasyon ($\rho > 0$) olduğunda her ikisi de yanlı sonuçlar üretir. Makale bu sorunu çözmek ve etkileşimin şeklini görselleştirmek için **REPID** (*Regional Effect Plots with implicit Interaction Detection*) yöntemini önerir: Odak özelliğin bireysel koşullu beklenti (**ICE**) eğrileri kendi ortalamaları etrafında merkezlenir ve diğer özellikler üzerinde $L_2$ heterojenlik kaybını (`SS_L2`) minimize eden bir karar ağacıyla alt bölgelere ayrılır. Ağacın her bölünmede sağladığı göreli risk azalması (`intImp`) etkileşen özellikleri doğru sıralarken, yaprak düğümlerindeki bölgesel PDP'ler (REP) etkileşimin yönünü açıkça görselleştirir.

---

## 4. Yeniden Üretim Yöntemi
- **R Kodunun Python'a Taşınma Doğrulaması:**
  - Makalenin resmi deposu (`JuliaHerbinger/REPID`) R dilindedir (`iml`, `ALEPlot`, `R6` paketleri). Çalışmamızda `R/simulations/generate_data_sim_weak.R` içindeki Bölüm 4.1 Gauss kopulalı veri üretim süreci ve `R/tree_splitting.R` içindeki merkezlenmiş ICE eğrilerini `SS_L2` kriteriyle bölen REPID algoritması **bire bir** Python/NumPy'a taşınmıştır.
  - R kodunda 120 rastgele permütasyonla yaklaşık hesaplanan Shapley etkileşimleri yerine, Python'da `XGBRegressor` üzerinde kesin C++ `shap.TreeExplainer(model).shap_interaction_values(X)` yöntemi kullanılmıştır.
- **Veri Setleri:**
  1. **Makale Bölüm 4.1 Simülasyon Verisi:** $n = 1.000$, Gauss kopulası ile üretilen $X_1, X_2, X_3, X_4 \sim \mathcal{U}(-1, 1)$, $y = r(\mathbf{x}) + \epsilon$ ($\epsilon \sim \mathcal{N}(0, (0.1\sigma(r(\mathbf{x})))^2)$).
  2. **Kendi Veri Setimiz (`veri.csv` — 1.92 MB):** California Housing (`sklearn.datasets.fetch_california_housing`, $n = 20.640$ gözlem, $p = 8$ özellik, hedef: `MedHouseVal`).
- **Sabit Tohum ve Ortam:** `SEED = 42`, `FAST = True`.
  - **Sürümler:** `Python 3.13.14`, `xgboost 3.4.1`, `shap 0.54.1`, `scikit-learn 1.9.1`, `scipy 1.18.1`, `numpy 2.5.3`, `pandas 3.0.6`, `matplotlib 3.11.2`, `seaborn 0.13.2`.
- **Çalıştırma Adımı:**
  ```bash
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 analiz.ipynb
  ```

---

## 5. Sonuç Tablosu ("Ne Yeniden Üretildi?")

| # | Makale İddiası / Bulgu | Durum | Kanıt (`analiz.ipynb` Hücre / Grafik / Sayısal Çıktı) | Açıklama |
|---|---|---|---|---|
| **1** | **Sıfırdan NumPy H-İstatistiği Birim Testleri (`assert`):** Toplamsal modelde $H^2=0$, saf çarpım modelinde $H^2=1$ çıkar. | **🟢 Yeniden Üretildi** | Hücre 5 `unit_df` Tablosu ve `assert` Testleri | Toplamsal $f=2.5x_1-1.8x_2$ için $H^2_{12}=0.000000$; sıfır ortalamalı $f=x_1 x_2$ için $H^2_{12}=1.000000$ ($H^*_{12}=0.3212$) doğrulanmıştır. |
| **2** | **Teorem 4 & Bölüm 4.1 (Ana Etki Yanılsaması / Sıralama Tersinmesi):** Friedman $H^2_{jk}$ paydası ana etkilere bağlıdır; daha zayıf bir etkileşimi 1. sıraya taşıyabilir. | **🟢 Yeniden Üretildi** | Hücre 8–9 Tabloları (`Kurgu A` & `Kurgu B`) ve 3'lü Sütun Grafiği | Gerçekte $\beta_{23}=1.6 > \beta_{12}=0.8$ (2 kat güçlü) iken, $\beta_3=3.5$ ana etkisi paydayı 13.8 kat büyüttüğü için ($4.4853$ vs $0.3248$) $H^2_{12}=0.1035 > H^2_{23}=0.0560$ ile yanlışlıkla 1. çıkmıştır. |
| **3** | **SHAP Etkileşim İndeksi ve $H^*_{jk}$'nin Ana Etkiden Bağımsızlığı:** SHAP etkileşim indeksi ($I_{j,l}$) ve normalize edilmemiş $H^*_{jk}$ ana etki değişiminden etkilenmez ve doğru sıralamayı korur. | **🟢 Yeniden Üretildi** | Hücre 8–9 `Kurgu B` Tablosu ve Grafiği | Hem $H^*_{jk}$ ($0.4985$ vs $0.1819$) hem de TreeSHAP etkileşim indeksi ($0.4111$ vs $0.1340$, bağıl $0.7181$ vs $0.2337$) gerçek $(X_2, X_3) > (X_1, X_2)$ sırasını %100 yakalamıştır. |
| **4** | **Korelasyon Tuzağı ($\rho \in \{0, 0.5, 0.9\}$):** Özellikler yüksek korelasyonlu olduğunda hem H-İstatistiği hem de SHAP etkileşim indeksi sahte etkileşim üretir veya eşit etkileşimlerin sırasını bozar. | **🟢 Yeniden Üretildi** | Hücre 12 `corr_summary` Tablosu ve Hücre 13 Çizgi Grafikleri | $\rho_{12}=0.9$'da gerçekte etkileşimsiz $(X_1, X_2)$ çiftinde $H^*_{12}$ %82 ($0.0464 \to 0.0844$), SHAP etkileşimi ise **3.7 kat** ($0.0429 \to 0.1600$) artmıştır. |
| **5** | **Boş (Null) Referans Kalibrasyonu (`max_depth=1`):** Eklemeli modelde $H=0$ çıkar; kısıtsız ağaçlarda sonlu örneklem gürültü tabanı oluşur. | **🟢 Yeniden Üretildi** | Hücre 16 `df_null_cal` Tablosu | `max_depth=1` modelinde $H_{jk}=0.0000$, `max_depth=4` modelinde etkileşimsiz çiftlerde $H \approx 0.032-0.045$ gürültü tabanı ölçülmüştür. |
| **6** | **Gerçek Veri (California Housing) Sıralama Farkı ve 2B PDP:** $H^2_{jk}$ paydası küçük çiftleri şişirirken, $H^*_{jk}$ ile SHAP yüksek sıralama uyumu gösterir. | **🟢 Yeniden Üretildi** | Hücre 19–20 Tabloları, Saçılım Grafiği ve 2B PDP Isı Haritası | 28 çiftte standart $H^2_{jk}$ ile SHAP arasında Spearman $\rho_S = 0.1346$ ($p=0.49$) iken, $H^*_{jk}$ ile SHAP arasında **$\rho_S = 0.9365$** ($p=2.42 \times 10^{-13}$) uyum bulunmuştur! |
| **7** | **REPID Bölgesel PDP ve `intImp` (BONUS):** ICE eğrilerinin karar ağacıyla bölünmesi hem etkileşimi sıralar hem de bölgesel etkiyi görselleştirir. | **🟢 Yeniden Üretildi** | Hücre 27–28 `fit_repid_python` Tablosu ve Bölgesel PDP Grafiği | 4 yaprak bölge ile `Latitude` ICE heterojenliği **%62.22** azaltılmış ve `Longitude` ($\text{intImp}=0.6222$) 1. etkileşen özellik olarak bulunmuştur. |

---

## 6. Sınırlar ve Eleştirel Değerlendirme
1. **Payda Kararsızlığı ve Sahte Sıralama (Hücre 19 Tablo 4):** California Housing verisinde `AveBedrms × Population` çiftinin 2B PD payda varyansı yalnızca $0.0028$'dir. Mutlak etkileşimde 28 çift arasında en sonuncu (**28. sıra**, $\text{SHAP}=0.0077, H^*=0.0189$) olmasına rağmen, paydası sıfıra yakın olduğu için standart Friedman $H^2_{jk}=0.1290$ ile **3. sıraya** fırlamıştır! Aynı şekilde `HouseAge × Population` mutlak etkileşimde **27. sıradayken** standart $H^2_{jk}$'de **5. sıraya** yerleşmiştir. Buna karşın `MedInc × AveOccup` gibi gerçekte 2. sırada olan güçlü bir etkileşim, `MedInc`'in yüksek ana etkisi paydayı ($0.4507$) büyüttüğü için $H^2_{jk}$'de 9. sıraya düşmüştür.
2. **$\mathcal{O}(n^2)$ Hesaplama Maliyeti ve Alt-Örnekleme Varyansı (Hücre 23–24):** Gözlem tabanlı ikili $H^2_{jk}$ tam $2n^2$, toplam $H^2_j$ ise $3n^2$ model tahmini gerektirir ($n=480$ için tek bir özellikte $691.200$ tahmin çağrısı!). Hızlandırmak için $n=80$ veya $n=25$ gibi küçük alt-örneklemler çekildiğinde ise 20 tekrarlı kutu grafiğimizde görüldüğü üzere tahmin varyansı patlamakta ($n=25$ için $H = 0.672 \pm 0.105$, aralık $[0.531, 0.917]$) ve sıralama kararsızlaşmaktadır.
3. **Model Derinliğine (`max_depth`) Aşırı Duyarlılık (Hücre 24 Panel 3):** XGBoost `max_depth` parametresi 2'den 7'ye çıkarıldığında, `AveRooms × AveBedrms` gibi eşdoğrusal çiftlerin $H_{jk}$ değeri ağaçların aşırı uyum (*overfitting*) bölmeleri nedeniyle hızla artmaktadır.
4. **REPID'in Sınırı:** REPID tek bir odak özelliğin ($x_S$) diğer özelliklerle etkileşimini bölgesel olarak mükemmel açıklasa da, tüm $\binom{p}{2}$ çiftlerini aynı anda tek bir küresel matriste özetlemek için her özellik için ayrı bir ICE ağacı kurulmasını gerektirir.

---

## 7. Kitapla Bağlantı (Molnar — Bölüm 21: Feature Interaction)
- **Bölüm 21.2 (Theory: Friedman's H-statistic):** Kitaptaki Denklem (21.2) ikili $H^2_{jk}$ ve Denklem (21.3) toplam $H^2_j$ formülleri Hücre 5'te hiçbir hazır kütüphane kullanmadan doğrudan NumPy ile kodlanmış ve birim testleriyle doğrulanmıştır.
- **Bölüm 21.4 (Disadvantages — Sınırlılıklar):**
  1. *"H-statistic is slow to compute ($2n^2$ / $3n^2$) and unstable when sampled":* Hücre 23 ve 24'te süre tablosu ve 20 tekrarlı kutu grafiğiyle kanıtlanmıştır.
  2. *"No model-agnostic statistical test / unclear threshold":* Hücre 16'da `max_depth=1` eklemeli (null) model kalibrasyonu ile çözülmüştür.
  3. *"Correlated features lead to spurious interactions":* Hücre 12–13'te $\rho \in \{0.0, 0.5, 0.9\}$ deneyiyle kanıtlanmıştır.
  4. *"Spurious interactions when denominator is small ($H^*$ alternative — Inglis et al., 2022)":* Makalemizin Teorem 4 bulgusu, kitaptaki bu sınırlılığın tam teorik temelidir ve Hücre 8 ile Hücre 19'da $H^*_{jk}$ ve SHAP etkileşim indeksiyle kıyaslanarak doğrulanmıştır.
  5. *"H does not tell us what the interaction looks like — inspect 2D PDP":* Hücre 20'de en güçlü çift (`Latitude × Longitude`) için 2B PDP ısı haritası ve Hücre 28'de REPID bölgesel PDP grafikleri çizilmiştir.

---

## 8. SAVUNMA HAZIRLIĞI (Canlı Sunum Rehberi)

### 8.1. 5 Dakikalık Canlı Akış Planı
1. **Dakika 0:00 – 0:50 (Teorem 4 & Sıfırdan NumPy Doğrulaması — Hücre 1 & 5):**
   - *Ne Gösterilecek:* Sıfırdan yazdığımız NumPy H-İstatistiği kodu ve `assert` doğrulama tablosu (Toplamsal $H^2=0$, Saf çarpım $H^2=1$).
   - *Ne Söylenecek:* "Hocam, Bölüm 21'deki Friedman H-İstatistiğini sıfırdan NumPy ile yazdık ve birim testlerinden geçirdik. Seçtiğim AISTATS 2022 (REPID) makalesi, H-İstatistiğinin paydasının tekil ana etkilere bağlı olduğunu (Teorem 4) ve bu yüzden en yüksek H skorunun en güçlü etkileşim olmayabileceğini kanıtlıyor."
2. **Dakika 0:50 – 2:00 (Makale Bölüm 4.1 Simülasyonu: Sıralama Tersinmesi — Hücre 8 & 9):**
   - *Ne Gösterilecek:* Kurgu B tablosu ve 3'lü sütun grafiği ($H^2_{jk}$ vs $H^*_{jk}$ vs SHAP Etkileşim İndeksi).
   - *Ne Söylenecek:* "Kurgu B'de `(X2, X3)` çiftinin gerçek etkileşimi ($\beta_{23}=1.6$), `(X1, X2)` çiftinin etkileşiminden ($\beta_{12}=0.8$) tam 2 kat daha güçlüdür. Ancak $X_3$'ün ana etkisi büyük ($\beta_3=3.5$) olduğu için paydası 13.8 kat şişiyor ve standart Friedman $H^2_{jk}$ sıralamayı tersine çevirerek zayıf `(X1, X2)` çiftini 1. sıraya koyuyor! Payda bölmesini kaldıran $H^*_{jk}$ ve ana etkiden arındırılmış SHAP etkileşim indeksi ise gerçek 2:1 sırasını hatasız yakalıyor."
3. **Dakika 2:00 – 3:15 (Kendi Verimiz: California Housing'de Büyük Sürpriz — Hücre 19 & 20):**
   - *Ne Gösterilecek:* 28 özellik çiftinin Spearman uyum çıktısı ($\rho_S = 0.1346$ vs $\rho_S = 0.9365$), Sahte $H^2_{jk}$ tablosu ve 2B PDP ısı haritası.
   - *Ne Söylenecek:* "Teorem 4'ün gerçek verideki etkisini California Housing'de çok çarpıcı gördük: Standart $H^2_{jk}$ ile SHAP sıralaması arasındaki Spearman korelasyonu yalnızca $0.1346$! Çünkü `AveBedrms × Population` çifti mutlak etkileşimde 28 çift arasında sonuncu (28.) olmasına rağmen, paydası $0.0028$ olduğu için standart $H^2_{jk}$'de 3. sıraya fırlıyor. Normalize edilmemiş $H^*_{jk}$ kullandığımızda ise SHAP ile sıralama uyumu bir anda **$0.9365$'e** çıkıyor!"
4. **Dakika 3:15 – 4:15 (Korelasyon, Null Kalibrasyon ve $2n^2$ Sınırları — Hücre 12, 16, 23, 24):**
   - *Ne Gösterilecek:* $\rho \in \{0, 0.5, 0.9\}$ sahte etkileşim çizgi grafiği, `max_depth=1` null tablosu ve 20 tekrarlı örneklem kararsızlığı kutu grafiği.
   - *Ne Söylenecek:* "Özellikler $\rho=0.9$ korelasyonlu olduğunda gerçekte etkileşimi olmayan çiftte hem H hem SHAP sahte etkileşim üretiyor. Ayrıca $2n^2$ maliyeti yüzünden küçük alt-örneklem ($n=80$) alındığında H skorlarının kutu grafiğinde ciddi kararsızlık gösterdiğini doğruladık."
5. **Dakika 4:15 – 5:00 (BONUS: REPID Bölgesel PDP & Sonuç — Hücre 27, 28, 30):**
   - *Ne Gösterilecek:* Küresel ICE/PDP vs REPID 4 yapraklı Bölgesel Etki Grafiği (REP) ve 6 maddelik karar rehberi.
   - *Ne Söylenecek:* "Son olarak makalenin REPID algoritmasını Python'da uyguladık; `Latitude` ICE eğrilerini `Longitude` üzerinden 4 alt bölgeye bölerek ICE heterojenliğini %62.22 azalttık."

### 8.2. Beklenen 12 Jüri Sorusu ve Kısa Doğru Cevaplar
1. **Soru:** *Friedman'ın H-İstatistiği neden daha zayıf bir etkileşimi daha güçlü gösterebiliyor?*  
   **Cevap:** Çünkü $H^2_{jk}$'nin paydası 2B kısmi bağımlılığın toplam varyansıdır ($\text{Var}(\text{PD}_{jk})$) ve özelliklerin tekil ana etki varyanslarını da içerir. Ana etkileri güçlü olan bir çiftin paydası büyüdüğü için $H^2_{jk}$ oranı küçülür; ana etkileri sıfıra yakın olan bir çiftin ise paydası çok küçük olduğu için en ufak etkileşim (hatta gürültü) $H^2_{jk}$'yi 1'e doğru şişirir.
2. **Soru:** *Normalize edilmemiş $H^*_{jk}$ nedir ve bu sorunu nasıl çözer?*  
   **Cevap:** $H^*_{jk}$, $H^2_{jk}$ formülündeki paydayı kaldırıp yalnızca payın (etkileşim varyansının) karekökünü alır: $H^*_{jk} = \sqrt{\frac{1}{n}\sum [\text{PD}_{jk} - \text{PD}_j - \text{PD}_k]^2}$. Birimi doğrudan hedef değişkenin ($y$) birimidir ve ana etki büyüklüğünden hiç etkilenmez.
3. **Soru:** *California Housing verisinde $H^2_{jk}$ ile SHAP arasındaki Spearman korelasyonu $0.1346$ iken $H^*_{jk}$ ile SHAP arasında neden $0.9365$ çıktı?*  
   **Cevap:** Çünkü California Housing'de `MedInc` gibi özelliklerin ana etkisi çok büyükken, `AveBedrms` ve `Population` gibi özelliklerin ana etkisi çok küçüktür. Standart $H^2_{jk}$, küçük paydalı `AveBedrms × Population` çiftini 28. sıradan 3. sıraya sıçratarak sıralamayı bozarken, hem $H^*_{jk}$ hem de SHAP mutlak etkileşim büyüklüğünü ölçtüğü için %93.65 uyum göstermiştir.
4. **Soru:** *SHAP etkileşim indeksi ($I_{j,l}$) nasıl hesaplanır ve neden ana etkilerden etkilenmez?*  
   **Cevap:** Shapley etkileşim değeri $\Phi_{j,l}(\mathbf{x})$, oyun teorisindeki ayrık ikinci mertebe farktan ($\nabla_{j,l}(\mathbf{x}_S) = f(\mathbf{x}_{S\cup\{j,l\}}) - f(\mathbf{x}_{S\cup\{j\}}) - f(\mathbf{x}_{S\cup\{l\}}) + f(\mathbf{x}_S)$) türetilir. Tekil ana etkiler bu fark işleminde birbirini tam olarak götürür. Küresel indeks ise tüm gözlemler üzerinden mutlak ortalamadır: $I_{j,l} = \frac{1}{n}\sum_{i=1}^n |\Phi_{j,l}(\mathbf{x}^{(i)})|$.
5. **Soru:** *Özellikler arasında yüksek korelasyon ($\rho = 0.9$) olduğunda neden hem H-İstatistiği hem de SHAP etkileşim indeksi yanılıyor?*  
   **Cevap:** H-İstatistiği marjinal dağılım üzerinde ortalama aldığı için korelasyonlu veride gerçekte var olmayan $(x_j, x_k)$ kombinasyonlarında modeli sorgular (ekstrapolasyon). Ağaç tabanlı SHAP ise eğitim sırasında korelasyonlu özelliklerin birbirinin yerine bölünme yapması ve ortak yaprak yollarını paylaşması nedeniyle sahte etkileşim üretir.
6. **Soru:** *İkili $H^2_{jk}$ ile toplam $H^2_j$ arasındaki hesaplama maliyeti farkı nedir?*  
   **Cevap:** $n$ gözlem üzerinden tam hesaplamada ikili $H^2_{jk}$ için $\text{PD}_j$, $\text{PD}_k$ ve $\text{PD}_{jk}$ hesaplanırken $\mathcal{O}(2n^2)$ tahmin çağrısı gerekir; toplam $H^2_j$ için ise $\text{PD}_j$ ve $\text{PD}_{-j}$ hesaplanırken $\mathcal{O}(3n^2)$ tahmin çağrısı gerekir.
7. **Soru:** *H-İstatistiğinin $1.0$'dan büyük çıkması mümkün müdür?*  
   **Cevap:** Evet! Molnar Bölüm 21'de belirtildiği üzere, iki özelliğin ortak varyansı tekil varyansların toplamından küçük olduğunda (veya Monte Carlo alt-örnekleme gürültüsüyle) $H^2_{jk} > 1$ çıkabilir.
8. **Soru:** *Boş (null) referans olarak neden `max_depth=1` XGBoost modelini kullandınız?*  
   **Cevap:** `max_depth=1` olan bir karar ağacı (*decision stump*) yalnızca tek bir özellik üzerinde bölünme yapabilir; dolayısıyla ağaç topluluğu tamamen toplamsaldır ($f(\mathbf{x}) = \sum_j f_j(x_j)$) ve teorik etkileşimi tam sıfırdır. Bu modelde $H_{jk}=0$ olduğunu doğrulamak ve kısıtsız modeldeki gürültü tabanıyla ($H \approx 0.03-0.05$) karşılaştırmak pratik bir kalibrasyon sağlar.
9. **Soru:** *REPID algoritması etkileşimi nasıl tespit ediyor ve bölmeyi neye göre yapıyor?*  
   **Cevap:** Odak özellik $x_S$'nin her gözlem için ICE eğrisini çıkarıp kendi ortalamasından merkezler ($Y_c = Y - \text{rowMeans}(Y)$). Eğer etkileşim yoksa tüm merkezlenmiş ICE eğrileri üst üste biner. Etkileşim varsa eğriler farklılaşır; REPID diğer özellikler ($X_{-S}$) üzerinde bir karar ağacı kurarak her düğümde eğrilerin bölgesel PDP'ye olan $L_2$ uzaklığını (`SS_L2`) minimize eder.
10. **Soru:** *Makalenin R kodunu Python'a taşırken bire bir taşıdığınız ve uyarladığınız kısımlar nelerdir?*  
    **Cevap:** Bölüm 4.1'deki Gauss kopulalı veri üretim fonksiyonunu (`generate_data_sim_weak.R`), H-İstatistiği formüllerini ve REPID'in merkezlenmiş ICE üzerinde `SS_L2` karar ağacı bölmesi ile `intImp` hesabını (`tree_splitting.R`) bire bir taşıdık. R kodundaki permütasyon tabanlı yaklaşık Shapley etkileşimi yerine ise `XGBRegressor` üzerinde kesin C++ `TreeSHAP` etkileşim tensörünü kullandık.
11. **Soru:** *H-İstatistiği yüksek çıkan bir çiftin nasıl etkileştiğini nereden anlarız?*  
    **Cevap:** H-İstatistiği tek bir skaler sayıdır, etkileşimin yönünü veya şeklini göstermez. Bunun için Hücre 20'de yaptığımız gibi **2 Boyutlu PDP (2B PDP)** ısı haritası veya Hücre 28'deki gibi **REPID Bölgesel PDP** grafikleri çizilmelidir.
12. **Soru:** *Uygulamada bir veri bilimciye etkileşim analizi için tek cümlelik öneriniz ne olur?*  
    **Cevap:** Standart $H^2_{jk}$'yi asla tek başına kullanmayın; mutlak etkileşim sırası için mutlaka **normalize edilmemiş $H^*_{jk}$** veya **TreeSHAP etkileşim indeksi** ile birlikte raporlayın ve yüksek korelasyon varsa **REPID / ALE / TreeHFD** ile doğrulayın.

### 8.3. Emin Olmadığım / Açık Kalan Noktalar
- Makalenin Bölüm 4.1 R kodunda (`helper_simulation.R`), doğrusal veri üretim süreci üzerinde tanımsız etkileşimlerden kaçınmak için bazı ayarlarda etkileşimli doğrusal model (`lm(y ~ .^2)`) kullanılırken Bölüm 4.2'de GAM ve XGBoost kullanılmıştır. Biz görev tanımımız gereği Bölüm 4.1 simülasyonunda doğrudan `XGBRegressor` eğittik ve Kurgu A'da ağaçların sonlu örneklemde $\beta_1=0.1$ özelliğini daha az bölmesi nedeniyle $H^*_{12}$'nin $H^*_{23}$'ten hafifçe küçük ($0.2597$ vs $0.2958$) çıktığını, ancak $H^2_{12}$'nin ($0.1882 > 0.1326$) yine de makaledeki gibi ters yönde şiştiğini açıkça gözlemledik ve dürüstçe raporladık.

### 8.4. Canlıda Hata Çıkarsa B Planı
1. **Veri Dosyası (`veri.csv`) Sorunu:** `veri.csv` (1.92 MB) klasörde yerel olarak mevcuttur; silinirse notebook `sklearn.datasets.fetch_california_housing` (o da olmazsa `load_diabetes`) üzerinden otomatik yedekleme yapabilecek yapıdadır.
2. **Süre Sorunu ($2n^2$ H Hesabı):** `FAST = True` bayrağı sayesinde 28 çiftin H-İstatistiği ve 15 tekrarlı simülasyon sonuçları sistem geçici klasöründen (`tempfile.gettempdir()`) <1 saniyede yüklenir. Önbellek silinse bile örneklem boyutları ($n_{\text{eval}} = 150 - 220$) optimize edildiği için tüm notebook sıfırdan ~35 saniyede tamamlanmaktadır.
3. **Çekirdek Sorunu:** `analiz.ipynb` tüm çıktıları ve grafikleri gömülü olarak kaydedilmiştir (`0.82 MB`).

---

## 9. Kaynakça
1. **Herbinger, J., Bischl, B., & Casalicchio, G. (2022).** *REPID: Regional Effect Plots with implicit Interaction Detection*. Proceedings of The 25th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 151:10209–10233. https://arxiv.org/abs/2202.07254
2. **Molnar, C. (2022).** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (Chapter 21: Feature Interaction). https://christophm.github.io/interpretable-ml-book/interaction.html
3. **Friedman, J. H., & Popescu, B. E. (2008).** *Predictive learning via rule ensembles*. The Annals of Applied Statistics, 2(3), 916–954.
4. **Inglis, A., Parnell, A., & Hurley, C. B. (2022).** *Visualizing Variable Importance and Variable Interaction Effects in Machine Learning Models*. Journal of Computational and Graphical Statistics, 31(3), 766–778.
5. **Greenwell, B. M., Boehmke, B. C., & McCarthy, A. J. (2018).** *A Simple and Effective Model-Based Variable Importance Measure*. arXiv preprint arXiv:1805.04755.
6. **Lundberg, S. M., Erion, G. G., & Lee, S. I. (2018).** *Consistent Individualized Feature Attribution for Tree Ensembles*. arXiv preprint arXiv:1802.03888.
