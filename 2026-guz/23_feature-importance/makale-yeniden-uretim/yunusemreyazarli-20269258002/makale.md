# Akademik Makale Yeniden Üretimi (Reproducibility): Kısıtsız Permütasyon Ekstrapolasyona Zorlar — PFI Neden Çöker ve LOFO / Koşullu Yöntemler Bunu Nasıl Çözer?

**Hazırlayan:** Yunus Emre Yazarlı  
**Öğrenci No:** 20269258002  
**Ders:** Açıklanabilir Yapay Zeka (XAI) — 2026 Güz  
**İlgili Kitap Bölümleri:** [Bölüm 23: Permutation Feature Importance (PFI)](https://christophm.github.io/interpretable-ml-book/feature-importance.html) & [Bölüm 24: Leave One Feature Out (LOFO) Importance](https://christophm.github.io/interpretable-ml-book/lofo.html)  
**Yeniden Üretilen Makale:** Giles Hooker, Lucas Mentch ve Siyu Zhou (2021). *"Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance"* (İlk preprint başlığı: *"Please Stop Permuting Features: An Explanation and Alternatives"*). **Statistics and Computing**, 31(6), Makale 82. ([arXiv:1905.03151](https://arxiv.org/abs/1905.03151) | [ResearchGate](https://www.researchgate.net/publication/355755411_Unrestricted_permutation_forces_extrapolation_variable_importance_requires_at_least_one_more_model_or_there_is_no_free_variable_importance) | [Semantic Scholar](https://www.semanticscholar.org/paper/Please-Stop-Permuting-Features%3A-An-Explanation-and-Hooker-Mentch/e6121c3744b9af235b32d35e87350ffd8b390efd))

## 1. Makale Seçim Gerekçesi ve XAI Dersiyle Bağlantısı

Bu çalışma, ödev yönergesindeki **Seçenek 1 (Zafiyet / Çöküş Analizi)** ve **Seçenek 2 (Algoritmalar Arası Kıyaslama / Benchmarking)** kriterlerini aynı anda karşılamaktadır:

* **Bölüm 23 (PFI) Zafiyeti:** [Breiman (2001)](https://doi.org/10.1023/A:1010933404324) ve [Fisher, Rudin ve Dominici (2019)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8323609/) tarafından popülerleştirilen *Permute-and-Predict (Karıştır ve Tahmin Et - PaP)* yaklaşımı, yeni bir model eğitmeyi gerektirmediği için hesaplama açısından ucuzdur. Ancak [Hooker, Mentch ve Zhou (2021)](https://arxiv.org/abs/1905.03151), özellikler arasında istatistiksel bağımlılık (korelasyon) bulunduğunda kısıtsız permütasyonun ($VI_j^\pi$) esnek modelleri (Random Forest, Yapay Sinir Ağları) **eğitim verisinin hiç bulunmadığı imkansız bölgelerde (ekstrapolasyon bölgesinde)** tahmine zorladığını ve korelasyonlu özelliklerin önemini yapay olarak şişirdiğini kanıtlamaktadır.
* **Bölüm 24 (LOFO) ve Koşullu PFI Çözümü:** Makale, güvenilir bir özellik önemi ölçümü için *"bedava özellik önemi yoktur (there is no free variable importance)"* ilkesini ortaya koymakta ve ekstrapolasyonu önlemek için **en az bir ek model** gerektiğini göstermektedir: ya özellik çıkarılıp ana model yeniden eğitilmelidir (**Dropped Variable Importance / LOFO - Bölüm 24**, [Lei vd. 2018](https://doi.org/10.1080/01621459.2017.1307116)) ya da koşullu dağılım $P(X_j \mid X_{-j})$ modellenmelidir (**Conditional PFI - Bölüm 23.3**, [Strobl vd. 2008](https://doi.org/10.1186/1471-2105-9-307)).

## 2. Matematiksel Çerçeve ve Makalenin 3 Temel Teoremi

### 2.1 Simülasyon Veri Üretim Süreci — Denklem (2)

[Hooker, Mentch ve Zhou (2021)](https://arxiv.org/abs/1905.03151) Bölüm 2'de, gerçek özellik önem sıralamasının analitik olarak bilindiği $p = 10$ özellikli doğrusal veri üretim modelini tanımlar:

$$y_i = 1.0 x_{i1} + 1.0 x_{i2} + 1.0 x_{i3} + 1.0 x_{i4} + 1.0 x_{i5} + 0 x_{i6} + 0.5 x_{i7} + 0.8 x_{i8} + 1.2 x_{i9} + 1.5 x_{i10} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 0.1^2)$$

* Her bir $x_{ij}$ özelliği marjinal olarak $[0, 1]$ aralığında Tekdüze ($\text{Uniform}(0, 1)$) dağılıma sahiptir.
* $x_3, \dots, x_{10}$ özellikleri birbirinden tamamen bağımsız üretilirken, **yalnızca ilk iki özellik $(x_1, x_2)$ bir Gaussian Copula aracılığıyla $\rho$ korelasyon parametresiyle** birbirine bağlanır:
  $$(Z_1, Z_2) \sim \mathcal{N}\left(\begin{bmatrix}0\\0\end{bmatrix}, \begin{bmatrix}1 & \rho \\ \rho & 1\end{bmatrix}\right), \quad x_{i1} = \Phi(Z_{i1}), \quad x_{i2} = \Phi(Z_{i2})$$
* Tüm özellikler aynı ölçekte ($\text{Var}(x_j) = 1/12$) olduğundan, gerçek önem sıralaması (1 = en düşük, 10 = en yüksek) doğrudan $|\beta_j|$ katsayı büyüklükleriyle belirlenir:
  * En önemsiz: $x_6$ ($\beta_6 = 0$, **Sıra: 1**), ardından $x_7$ ($\beta_7 = 0.5$, **Sıra: 2**) ve $x_8$ ($\beta_8 = 0.8$, **Sıra: 3**).
  * Eşit önemdeki referans grubu: $x_1, x_2, x_3, x_4, x_5$ ($\beta_1 = \dots = \beta_5 = 1.0$, **Teorik Ortalama Sıra: 6.0**).
  * En önemli iki özellik: $x_9$ ($\beta_9 = 1.2$, **Sıra: 9**) ve $x_{10}$ ($\beta_{10} = 1.5$, **Sıra: 10**).

### 2.2 Teorem 1 — Doğrusal Modeller Neden Yanılmazken Esnek Modeller Yanılır?

En küçük kareler (OLS) ile tahmin edilen doğrusal bir model $f(x) = \hat{\beta}_0 + \sum_{j=1}^p \hat{\beta}_j x_j$ için kısıtsız permütasyon öneminin ($VI_j^\pi$) beklenen değeri ([Gregorutti vd. 2015](https://doi.org/10.1016/j.csda.2015.04.002); [Fisher, Rudin ve Dominici 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8323609/)):

$$\mathbb{E}_\pi\left[VI_j^\pi\right] = 2 \hat{\beta}_j^2 \sum_{i=1}^N (x_{ij} - \bar{x}_j)^2$$

Doğrusal model tüm uzayda sabit $\hat{\beta}_j$ eğimiyle hareket ettiği için korelasyondan ($\rho$) etkilenmez. Ancak **Random Forest** ve **Yapay Sinir Ağları (YSA)** eğitim verisinin olmadığı köşelerde (örneğin $x_1 = 0.95, x_2 = 0.05$) yaprak ortalamalarını (*Potential Nearest Neighbors - pNN*) veya doğrusal olmayan salınımları (*Gibbs etkisi*) kullandığı için $\rho$ arttıkça $x_1$ ve $x_2$'nin PFI değerini aşırı şişirir.

### 2.3 Teorem 2 ve Teorem 3 — 4 Alternatif Yöntemin Eşdeğerliği (LOFO ve Koşullu PFI)

[Hooker, Mentch ve Zhou (2021)](https://arxiv.org/abs/1905.03151) Bölüm 5'te ekstrapolasyonu önleyen dört yöntemi tanımlar ve karesel kayıp ($L_2$) altında hepsinin aynı popülasyon hedefini tahmin ettiğini kanıtlar:

1. **Koşullu Permütasyon Önemi ($VI_j^C$ — Bölüm 23.3):** $x_{ij}^{c,j} \sim P(X_j \mid X_{-j})$ koşullu dağılımından örneklenerek $L(y_i, f(x_i^{c,j})) - L(y_i, f(x_i))$ hesaplanır.
2. **Özelliği Dışarıda Bırakma Önemi ($VI_j^D$ — Bölüm 24 LOFO / LOCO):** $j$. özellik veri setinden çıkarılır, $f_{-j}(x_{-j})$ modeli yeniden eğitilir ve $L(y_i, f_{-j}(x_{i,-j})) - L(y_i, f(x_i))$ hesaplanır.
3. **Karıştır ve Yeniden Eğit ($VI_j^{\pi L}$ — Permute-and-Relearn):** $j$. özellik karıştırıldıktan sonra $(y, X^{\pi,j})$ üzerinde yeni bir $f_{\pi,j}$ modeli eğitilir ([Mentch ve Hooker 2016](https://jmlr.org/papers/v17/14-168.html)).
4. **Koşullu Örnekle ve Yeniden Eğit ($VI_j^{CL}$ — Condition-and-Relearn):** $j$. özellik koşullu dağılımdan örneklenip yeni model $f_{c,j}$ eğitilir.

**Teorem 2 ve 3 Sonucu:** Doğrusal bağımlılık altında bu dört yöntemin beklenen değeri:

$$\mathbb{E}\left[VI_j^C\right] = 2\mathbb{E}\left[VI_j^D\right] = \mathbb{E}\left[VI_j^{\pi L}\right] = 2 \beta_j^2 \sum_{i=1}^N \text{Var}(x_{ij} \mid x_{i,-j}) = 2 \beta_j^2 (1 - \rho^2) \sum_{i=1}^N (x_{ij} - \bar{x}_j)^2$$

Yani korelasyon $\rho \to 1$'e yaklaştıkça, $x_2$ özelliği $x_1$'in bilgisini telafi ettiği (*compensation effect*) için LOFO ($VI_j^D$) ve Koşullu PFI ($VI_j^C$) değerleri $(1 - \rho^2)$ çarpanıyla **azalırken**, kısıtsız PFI ($VI_j^\pi$) tam tersine ekstrapolasyon nedeniyle **yapay olarak yükselir**!

## 3. Yeniden Üretim Deneyleri ve Bulgularımız (`analiz.ipynb` & `veri.csv`)

### Deney 1 (Makale Şekil 1 Yeniden Üretimi): $\rho = 0.0$ ve $\rho = 0.90$ İçin Önem Sıralamaları

Denklem (2) kurallarına göre üretilen veri setinde ($\rho = 0.0$ bağımsız ve $\rho = 0.90$ Gaussian Copula korelasyonlu) Random Forest, 20 gizli nöronlu Yapay Sinir Ağı (MLP) ve Doğrusal Regresyon (OLS) modelleri eğitilerek özelliklerin ortalama PFI sıralamaları (1 = en düşük, 10 = en yüksek) hesaplanmıştır:

| Özellik | Gerçek Katsayı ($\beta_j$) | Teorik Sıra | RF Sırası ($\rho=0.0$) | RF Sırası ($\rho=0.90$) | YSA Sırası ($\rho=0.90$) | Doğrusal Model ($\rho=0.90$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`x1` (Korelasyonlu)** | `1.0` | `6.0` | `6.00` | **`8.17` (Şişti ↑)** | **`7.83` (Şişti ↑)** | `5.50` |
| **`x2` (Korelasyonlu)** | `1.0` | `6.0` | `6.33` | **`8.67` (Şişti ↑)** | **`8.00` (Şişti ↑)** | `6.50` |
| **`x3` (Bağımsız)** | `1.0` | `6.0` | `5.17` | `4.67` | `5.33` | `6.67` |
| **`x4` (Bağımsız)** | `1.0` | `6.0` | `6.33` | `4.67` | `5.17` | `5.17` |
| **`x5` (Bağımsız)** | `1.0` | `6.0` | `5.50` | `5.17` | `5.50` | `6.17` |
| **`x6` (Gürültü)** | `0.0` | `1.0` | `1.00` | `1.00` | `1.00` | `1.00` |
| **`x7` (Zayıf)** | `0.5` | `2.0` | `2.00` | `2.00` | `2.00` | `2.00` |
| **`x8` (Orta)** | `0.8` | `3.0` | `3.83` | `3.50` | `3.17` | `3.00` |
| **`x9` (Güçlü)** | `1.2` | `9.0` | `8.83` | **`7.67` (Geçildi ↓)** | `8.50` | `9.00` |
| **`x10` (En Güçlü)** | `1.5` | `10.0` | `10.00` | `9.50` | `9.83` | `10.00` |

**Bulguların Yorumu:** Makaledeki Şekil 1 ile birebir uyumlu olarak, $\rho = 0.0$ durumunda tüm modeller gerçek sıralamayı yakalarken; $\rho = 0.90$ olduğunda Random Forest PFI'da $x_1$ (`8.17`) ve $x_2$ (`8.67`) özellikleri kendileriyle aynı katsayıya sahip $(x_3, x_4, x_5)$ grubunu ezici biçimde geçmekte, hatta kendilerinden daha büyük katsayıya sahip **$x_9$ ($\beta_9 = 1.2$, sıra `7.67`) özelliğinin bile üstüne çıkmaktadır**!

### Deney 2 (Makale Şekil 2 & Teorem 2–3 Yeniden Üretimi): Korelasyon ($\rho$) Taraması

Korelasyon katsayısı $\rho \in \{0.00, 0.15, 0.35, 0.55, 0.75, 0.90, 0.96\}$ boyunca artırıldığında $x_1$ özelliğinin ($\beta_1 = \beta_2 = 0.8$) hesaplanan önem değerleri ($\Delta\text{MSE}$):

| Korelasyon ($\rho$) | Kısıtsız PFI ($VI_1^\pi$ - Bölüm 23) | Koşullu PFI ($VI_1^C$ - Bölüm 23.3) | LOFO / Drop-and-Retrain ($VI_1^D$ - Bölüm 24) | Doğrusal Model PFI |
| :---: | :---: | :---: | :---: | :---: |
| **`0.00`** | `0.0470` | `0.0455` | `0.0079` | `0.1065` |
| **`0.35`** | `0.0560` | `0.0483` | `0.0088` | `0.1052` |
| **`0.75`** | `0.0801` | `0.0389` | `0.0041` | `0.1080` |
| **`0.90`** | `0.0758` | `0.0223` | `-0.0005` | `0.1049` |
| **`0.96`** | **`0.1002` (+113% Şişme)** | **`0.0145` (-68% Azalma)** | **`-0.0005` (~0.00'a İndi)** | `0.1061` |

### Deney 3 (Makale Bölüm 4, Şekil 4 Yeniden Üretimi): $y = x_1 + 0\cdot x_2$ Modelinde Ekstrapolasyon ve Sahte Önem

Gerçek veri üretim fonksiyonu $y = 1.0 x_1 + 0.0 x_2 + \epsilon$ ($\epsilon \sim \mathcal{N}(0, 0.05^2)$, $\rho = 0.92$) olduğunda $x_2$ özelliğinin hedef değişken $y$ üzerinde **hiçbir gerçek etkisi yoktur ($\beta_2 = 0$)**. Ancak eğitim verisi diyagonal etrafında toplandığı için Random Forest ağaçları diyagonal boyunca hem $x_1$ hem $x_2$ üzerinden bölünmeler yapar:

* **Kısıtsız PFI ($VI_2^\pi$):** $x_2$ karıştırıldığında sorgu noktaları sol-üst ve sağ-alt boş köşelere düşer; ağaç yaprakları bu köşelerde yanlış komşuları (*pNN*) ortaladığı için tamamen etkisiz olan $x_2$'ye **`0.0229 ΔMSE`** (gerçek sinyal $x_1$'in neredeyse %30'u kadar!) **sahte önem** verir.
* **LOFO ($VI_2^D$):** $x_2$ veri setinden çıkarılıp model yalnızca $x_1$ ile yeniden eğitildiğinde hata artışı **`0.0001 ΔMSE` (sıfır)** olur ve $x_2$'nin gereksiz bir özellik olduğu hatasız tespit edilir.

### Deney 4 (Makale Bölüm 5.2, Şekil 9 Yeniden Üretimi): Gerçek Dünya Verisi (`bike.csv`)

Makalenin 5.2 bölümünde kullanılan bisiklet kiralama veri seti (`bike.csv`) üzerinde Kısıtsız PFI ($VI_j^\pi$), Karıştır ve Yeniden Eğit ($VI_j^{\pi L}$) ve LOFO ($VI_j^D$) karşılaştırılmıştır:

| Özellik | Kısıtsız PFI ($\Delta\text{RMSE}$) | Permute-and-Relearn ($\Delta\text{RMSE}$) | LOFO / Drop-and-Retrain ($\Delta\text{RMSE}$) | PFI Sırası | Relearn Sırası | LOFO Sırası |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cnt_2d_bfr`** | `881.90` | `428.95` | `407.88` | `8` | `8` | `8` |
| **`temp`** | **`355.13`** | **`87.09`** | **`99.40`** | `7` | `7` | `7` |
| **`hum`** | `139.33` | `62.38` | `55.53` | `6` | `4` | `6` |
| **`weather`** | **`102.00`** | **`39.98`** | **`45.27`** | **`5`** | **`2`** | **`3`** |
| **`season`** | `78.37` | `64.78` | `50.87` | `4` | `5` | `5` |
| **`windspeed`** | `58.69` | `67.41` | `47.53` | `3` | `6` | `4` |
| **`workday`** | `20.46` | `47.06` | `39.62` | `2` | `3` | `2` |
| **`holiday`** | `0.20` | `20.55` | `22.54` | `1` | `1` | `1` |

Gerçek bisiklet verisinde birbiriyle yüksek korelasyonlu olan meteorolojik değişkenlerde (`temp`, `hum`, `weather`, `season`), kısıtsız PFI ekstrapolasyon (ör. kış mevsiminde 35°C sıcaklık veya %95 nemde güneşli hava kombinasyonları) nedeniyle hata artışını 3–4 kat şişirirken (`temp` için `355.13` vs LOFO'da `99.40`), yeniden eğitim tabanlı yöntemler (`LOFO` ve `Permute-and-Relearn`) her özelliğin yalnızca diğer özellikler tarafından açıklanamayan benzersiz katkısını ölçmektedir.

## 4. Klasördeki Dosyalar ve Kullanım Kılavuzu

* **`makale.md`:** Makalenin teorik çerçevesini, teoremlerini ve yeniden üretim bulgularını özetleyen bu akademik rapor.
* **`veri.csv`:** Makalenin Denklem (2) kurallarına göre üretilmiş ($N=1000, p=10, \rho=0.90$) Gaussian Copula simülasyon veri seti.
* **`analiz.ipynb`:** Tüm deneylerin Python kodlarını, konsol çıktılarını ve üretilen grafikleri içeren çalıştırılmış Jupyter Notebook dosyası.
* **`makale_python_kod_ve_ciktilar.html`:** Tarayıcıda doğrudan açılabilen, kopyalanabilir Python kod hücreleri ve gömülü grafik çıktılarından oluşan rapor.
* **`makale_interaktif_web_uygulamasi.html`:** Korelasyon ($\rho$), örneklem büyüklüğü ($N$) ve algoritma seçimiyle PFI ekstrapolasyonunu ve LOFO karşılaştırmasını canlı animasyonla gösteren interaktif web uygulaması.

## Kaynakça

* Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
* Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *Journal of Machine Learning Research*, 20(177), 1–81. [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8323609/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8323609/)
* Hooker, G., Mentch, L., & Zhou, S. (2021). Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. *Statistics and Computing*, 31(6), 82. [https://arxiv.org/abs/1905.03151](https://arxiv.org/abs/1905.03151)
* Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. *Journal of the American Statistical Association*, 113(523), 1094–1111. [https://doi.org/10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116)
* Molnar, C. (2025). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (Bölüm 23 & 24). [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
* Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., & Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics*, 9(1), 307. [https://doi.org/10.1186/1471-2105-9-307](https://doi.org/10.1186/1471-2105-9-307)
