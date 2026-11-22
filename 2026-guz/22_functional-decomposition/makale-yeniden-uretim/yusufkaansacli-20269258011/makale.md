# Bölüm 22: Fonksiyonel Ayrıştırma — Makale Yeniden Üretimi ve Eleştirel Analiz Raporu

**Hazırlayan:** Yusuf Kaan Saçlı (`20269258011`)  
**Ders:** Açıklanabilir Yapay Zeka (XAI) — Güz 2026  
**İlgili Kitap Bölümü:** Christoph Molnar, *Interpretable Machine Learning*, Bölüm 22: Fonksiyonel Ayrıştırma (*Functional Decomposition*)

---

## 1. Makale Künyesi (Doğrulanmış)
- **Başlık:** *Tree Ensemble Explainability through the Hoeffding Functional Decomposition and TreeHFD Algorithm*
- **Yazar:** Clément Bénard (Thales cortAIx-Labs, Fransa)
- **Yıl ve Yayın Yeri:** *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*
- **Doğrulanmış Bağlantılar:**
  - **OpenReview (Makale PDF & Hakem Değerlendirmeleri):** https://openreview.net/forum?id=dRLWcpBQxS
  - **arXiv Ön Baskı:** https://arxiv.org/abs/2506.03892
  - **Resmi Python Paketi (PyPI):** https://pypi.org/project/treehfd/ (`treehfd == 1.4.3`)
  - **Resmi Kaynak Kodu (GitHub):** https://github.com/ThalesGroup/treehfd

---

## 2. Seçilen Tip ve Neden
- **Seçilen Tip:** **Seçenek 3 — Geliştirme / Modifikasyon** (Temel algoritmaya hız, hesaplanabilirlik ve bağımlı girdilerde kararlılık katan yeni varyant).
- **Seçim Gerekçesi:** Molnar'ın 22. Bölümünde ele alınan temel algoritma olan **Fonksiyonel ANOVA (fANOVA)**, yüksek boyutlu bir kara kutu tahmin fonksiyonunu $f(\mathbf{x}) = f_0 + \sum_j f_j(x_j) + \sum_{j<k} f_{jk}(x_j, x_k) + \dots$ şeklinde ana etkilere ve etkileşimlere benzersiz (*unique*) biçimde ayırır; ancak özelliklerin **birbirinden bağımsız** olduğunu varsayar. Özellikler korelasyonlu olduğunda Hooker (2007) tarafından önerilen **Genelleştirilmiş fANOVA (Hoeffding Ayrıştırması)** hiyerarşik diklik (*hierarchical orthogonality*) koşuluyla benzersizliği korusa da, Molnar'ın kitapta açıkça belirttiği gibi çok değişkenli yoğunluk tahmini ve yüksek boyutlu integraller gerektirdiği için pratikte hesaplanması neredeyse imkânsız kalmıştır. Bénard (2025) tarafından geliştirilen **TreeHFD** algoritması, ağaç topluluklarının (*tree ensembles*) Kartezyen bölümleme yapısını kullanarak bu çözülemez integral problemini her ağaç için hızlı bir **Ağırlıklı En Küçük Kareler (*Weighted Least Squares*)** problemine indirgemekte ve bağımlı girdilerde hiyerarşik dik Hoeffding ayrıştırmasını saniyeler içinde hesaplanabilir kılmaktadır.

---

## 3. Makalenin 1 Sayfalık Sade Özeti
Tablosal verilerde en yüksek tahmin başarısını genellikle **XGBoost** ve **Rastgele Ormanlar** (*Random Forests*) gibi karar ağacı toplulukları sağlar. Ancak yüzlerce derin ağacın toplamından oluşan bu modeller birer "kara kutu" (*black-box*) gibi davranır. Bir modeli açıklamanın en güçlü matematiksel yolu, onun tahmin yüzeyini tek değişkenli **ana etki eğrileri** ($f_j(x_j)$) ve iki değişkenli **etkileşim yüzeylerinin** ($f_{jk}(x_j, x_k)$) toplamı olarak yazmaktır (**Fonksiyonel Ayrıştırma / Hoeffding Ayrıştırması**).

Ne var ki gerçek dünya verilerinde girdi özellikleri neredeyse her zaman birbiriyle bağımlıdır (örneğin Kaliforniya konut verisinde enlem ile boylam arasında $r = -0.9247$, bisiklet paylaşım verisinde sıcaklık ile hissedilen sıcaklık arasında $r = 0.9877$ korelasyon vardır). Girdiler korelasyonlu olduğunda, bir etkileşim terimi $f_{jk}(X_j, X_k)$ kendi alt bileşenleri olan $f_j(X_j)$ ve $f_k(X_k)$ ile kolayca karışır. Ayrıştırmanın benzersiz ve yorumlanabilir olması için **hiyerarşik diklik** (*hierarchical orthogonality*) kuralı sağlanmalıdır: Herhangi bir etkileşim bileşeni, içerdiği tekil ana etkilerle gerçek veri dağılımı üzerinde sıfır iç çarpıma (sıfır kovaryansa) sahip olmalıdır:
$$\mathbb{E}\left[f_j(X_j) \cdot f_{jk}(X_j, X_k)\right] = 0, \quad \forall j, k.$$

Sürekli fonksiyonlarda bu koşulu çözmek imkânsıza yakınken, Bénard (2025) şu kritik gözlemi yapar: Bir karar ağacı, girdi uzayını eksenlere paralel dikdörtgen kutulara böler. Dolayısıyla bir ağacın yaptığı tahmin, özelliklerin bölünme eşiklerinin oluşturduğu **Kartezyen ızgara hücreleri** (*Cartesian tree partitions*) üzerinde parçalı sabittir (*piecewise constant*). TreeHFD algoritması, her ağaç için sürekli hiyerarşik diklik integrallerini bu Kartezyen hücrelere düşen eğitim gözlemlerinin oranlarıyla (ampirik olasılık ağırlıklarıyla) ayrıklaştırır. Böylece her ağacın ana etki ve ikili etkileşim katsayıları, basit bir doğrusal en küçük kareler sistemi çözülerek elde edilir. Tüm ağaçların bileşenleri toplandığında:
1. Elde edilen ana etkiler ($\hat{f}_j(x_j)$) gürültüsüz, deterministik 1 boyutlu basamak fonksiyonlarıdır.
2. Yaygın kullanılan **TreeSHAP** yönteminde bir özelliğin ana etkisi ($\Phi_{j,j}(\mathbf{x})$) diğer özelliklerin değerlerine bağlı olarak gürültülü bir nokta bulutu oluştururken ve yüksek mertebeli etkileşimlerle karışırken, **TreeHFD** ana etkileri etkileşimlerden dik olarak ayırır.
3. Algoritma teorik olarak asimptotik tutarlılık (*convergence*), seyreklik (*sparsity*) ve nedensel değişken seçimi (*causal variable selection* — modelde yer almayan gürültü değişkenlerine sıfır etki atama) garantilerine sahiptir.

---

## 4. Yeniden Üretim Yöntemi
- **Veri Setleri:**
  1. **Makale Veri Seti (`veri_california.csv` — 1.92 MB):** `sklearn.datasets.fetch_california_housing` ($n = 20.640$ gözlem, $p = 8$ sayısal özellik, hedef: `MedHouseVal` [100.000 \$ biriminde medyan konut değeri]).
  2. **Kendi Veri Setimiz (`veri.csv` — 1.12 MB):** OpenML `Bike_Sharing_Demand` (v2, $n = 17.379$ saatlik gözlem, kodlanmış $p = 10$ özellik, hedef: `count` [saatlik kiralanan bisiklet sayısı]).
  3. **Kontrollü Simülasyon Verisi:** $f(x_1, x_2, x_3) = x_1 + \sin(x_2) + x_1 x_3 + \epsilon$ ($\epsilon \sim \mathcal{N}(0, 0.1^2)$, $n = 2.500$, çok değişkenli Gauss girdileri, çiftler arası korelasyon $\rho \in \{0.0, 0.5, 0.9\}$).
- **Model ve Hiperparametreler:** Makale ve resmi paket örneğiyle bire bir aynı `xgboost.XGBRegressor(eta=0.3, n_estimators=100, max_depth=5, random_state=42)`. Tam rekonstrüksiyon kontrolü için ayrıca `max_depth=2` modeli ve derinlik duyarlılığı için `max_depth` $\in \{2, 3, 5, 7\}$ modelleri eğitilmiştir.
- **Sabit Tohum ve Ortam:** `SEED = 42`, `FAST = True`.
  - **Sürümler:** `Python 3.13.14`, `treehfd 1.4.3`, `xgboost 3.4.1`, `shap 0.54.1`, `scikit-learn 1.9.1`, `numpy 2.5.3`, `pandas 3.0.6`, `matplotlib 3.11.2`, `seaborn 0.13.2`.
- **Çalıştırma Adımı:**
  ```bash
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 analiz.ipynb
  ```
  veya Jupyter Lab/Notebook üzerinden `Restart Kernel and Run All Cells` seçeneğiyle baştan sona çalıştırılır (`FAST=True` ile çalışma süresi ~15–25 saniye, önbelleksiz soğuk çalıştırmada ~95 saniyedir).

---

## 5. Sonuç Tablosu ("Ne Yeniden Üretildi?")

| # | Makale İddiası / Bulgu | Durum | Kanıt (`analiz.ipynb` Hücre / Grafik / Sayısal Çıktı) | Açıklama |
|---|---|---|---|---|
| **1** | **California Housing Boylam (`Longitude`) Ana Etkisinde SF Körfezi Tepesi:** TreeHFD $-122.4^\circ$ boylamındaki fiyat zirvesini net yakalar, TreeSHAP ise etkileşimlerle karıştığı için bulutlaşır. | **🟢 Yeniden Üretildi** | Hücre 12–16 Grafikleri ve SF Sıçrama Ölçümü | SF Körfezi ($-122.4^\circ$) vs iç bölge ($-121.0^\circ$) ana etki sıçraması TreeHFD'de **$+0.785$** iken TreeSHAP'ta **$+0.380$** düzeyinde kalmıştır (2 kat fark). |
| **2** | **Tam Rekonstrüksiyon (`max_depth=2`) ve Yüksek Sadakat (`max_depth=5`):** $\eta_0 + \sum \hat{f}_j + \sum \hat{f}_{jk}$ toplamı kara kutu tahminini korur. | **🟢 Yeniden Üretildi** | Hücre 12 `recon_table` ve Hücre 34 `df_depth` | `max_depth=2` için $\text{MAE} = 0.000001$ (**%100.00** sadakat); `max_depth=5` için 1B+2B bileşenler model varyansının **%97.99'unu** açıklamaktadır. |
| **3** | **Sıfır Ortalama ve Hiyerarşik Diklik ($\mathbb{E}[\hat{f}_j \hat{f}_{jk}] = 0$):** Ağaç hücreleri üzerindeki en küçük kareler çözümü diklik sağlar. | **🟢 Yeniden Üretildi** | Hücre 19 `prop_summary` Tablosu | Maksimum ortalama sapması $\le 2.47 \times 10^{-6}$ (`float32` hassasiyeti) ve ana etki-etkileşim iç çarpımı $3.58 \times 10^{-3}$ olarak doğrulanmıştır. |
| **4** | **Bağımlı Girdilerde Varyans Ayrıştırması:** Bileşen varyansları toplamı model varyansını verir. | **🟡 Kısmen** | Hücre 19 `prop_summary` (Satır 5) | $\hat{f}_j$ ile $\hat{f}_{jk}$ birbirine diktir; ancak $X_{\text{Lat}}$ ile $X_{\text{Lon}}$ korelasyonlu ($r=-0.9247$) olduğunda farklı ana etkiler arasında çapraz kovaryans ($\text{Cov}=-1.309$) bulunur (Bileşen Var Toplamı $=2.475$ vs Tahmin Var $=1.166$). |
| **5** | **Korelasyonlu Simülasyonda ($f = x_1 + \sin(x_2) + x_1 x_3$) Bileşen Geri Kazanımı:** Bağımlı girdilerde ($\rho=0.9$) TreeHFD analitik Hoeffding bileşenlerini TreeSHAP ve PDP'den daha doğru yakalar. | **🟢 Yeniden Üretildi** | Hücre 34 `df_sim` ve Hücre 35 Çizgi Grafiği | $\rho=0.9$'da TreeHFD; $f_2=\sin(x_2)$ ($\text{RMSE}=0.0790$), Hoeffding $f_1$ ($\text{RMSE}=0.0993$ vs SHAP $0.4415$) ve $f_{13}$ ($\text{RMSE}=0.0766$ vs SHAP $0.2854$) bileşenlerinde en düşük hatayı vermiştir. |
| **6** | **Kendi Verimiz (Bike Sharing Demand) Üzerinde Genellenebilirlik:** Saat (`hour`), sıcaklık (`temp`) ve iş günü (`workingday`) ayrıştırması. | **🟢 Yeniden Üretildi** | Hücre 22–23 Grafikleri ve Sadakat Ölçümü | `temp` ile `feel_temp` ($r=0.9877$) bağımlılığına rağmen model varyansının **%97.41'i** 1B+2B bileşenlerle açıklanmış (Test $R^2=0.9155$) ve `hour × workingday` etkileşimi ayrıştırılmıştır. |

---

## 6. Sınırlar ve Eleştirel Değerlendirme
Deneysel stres testlerimizde (Bölüm 9a–9e) `TreeHFD` algoritmasının ve `treehfd` paketinin şu 5 temel sınırı tespit edilmiştir:
1. **Dar Model ve Veri Tipi Desteği (Hücre 27):** `treehfd` yalnızca `xgboost.sklearn` (`XGBRegressor`, `XGBClassifier`) modellerini destekler; `scikit-learn` `RandomForestRegressor`, `LightGBM`, `CatBoost` veya Boosted Random Forest (`n_estimators > 1, num_parallel_tree > 1`) verildiğinde `ValueError` fırlatır. Ayrıca XGBoost'un yerel kategorik değişken desteğini (`enable_categorical=True`) ve doğrudan `pandas.DataFrame` girdisini kabul etmez; yalnızca `numpy.ndarray` ve one-hot/sayısal kodlama zorunludur.
2. **Çalışma Süresi Ölçeklenmesi (Hücre 30):** C++ içinde paralel çalışan `TreeSHAP` (`pred_interactions=True`) $n=20.000$ ve 300 ağaçta $5.32$ saniyede biterken, Python döngüsüyle ağaç başına en küçük kareler çözen `TreeHFD` $26.54$ saniye sürmektedir (~5 ile 13 kat daha yavaş).
3. **Derin Ağaçlarda ($\text{max\_depth} \ge 7$) Kartezyen Hücre Patlaması (Hücre 34):** Bir ağaçta derinlik arttıkça bölünen değişken çiftlerinin Kartezyen çarpımı büyür. `depth_variable` budama parametresi kullanılmazsa derin ağaçlarda en küçük kareler matrisi büyümekte ve çalışma süresi hızla artmaktadır.
4. **Bağımlı Girdilerde Çapraz Ana Etki Kovaryansı (Hücre 19):** Hiyerarşik diklik yalnızca alt küme-üst küme çiftleri ($\hat{f}_j$ ile $\hat{f}_{jk}$) arasında diklik sağlar; $X_j$ ve $X_k$ korelasyonlu olduğunda $\hat{f}_j(X_j)$ ile $\hat{f}_k(X_k)$ birbirine dik olamaz ($\text{Cov}(\hat{f}_j, \hat{f}_k) \neq 0$).
5. **Çarpımsal Terimlerin Karesel Ana Etki İndüklemesi (Hücre 34 — Hoeffding vs ALE Farkı):** $X_1$ ve $X_3$ korelasyonlu ($\rho > 0$) iken $X_1 X_3$ çarpımı, koşullu beklenti $\mathbb{E}[X_1 X_3 \mid X_1] = \rho X_1^2$ nedeniyle Hoeffding ayrıştırmasında $X_1$ üzerinde $\frac{\rho}{1+\rho^2}(X_1^2 - 1)$ şeklinde karesel bir ana etki indükler (`RMSE_f1_Struct_TreeHFD` $\rho=0.9$'da $0.6815$'e çıkar). Bu matematiksel olarak Hoeffding tanımına tam uygun olsa da (`RMSE_f1_Hoeff_TreeHFD = 0.0993`), yapısal doğrusal $x_1$ terimini görmek isteyen bir uygulayıcı için şaşırtıcı olabilir.

---

## 7. Kitapla Bağlantı (Molnar — Bölüm 22: Functional Decomposition)
- **Bölüm 22.2 (Why not just look at the formula?):** Molnar, $f(x_1, x_2) = x_1 x_2$ gibi saf bir çarpım fonksiyonunda bile özelliklerin dağılımı sıfır merkezli veya bağımsız değilse çarpım teriminin ana etki üreteceğini anlatır. Simülasyonumuzdaki (Hücre 34) $\frac{\rho}{1+\rho^2}(X_1^2 - 1)$ terimi, kitaptaki bu teorik uyarının bire bir analitik karşılığıdır.
- **Bölüm 22.4.2 (Generalized Functional ANOVA for Dependent Features — Hooker 2007):** Molnar bu alt başlıkta Hooker (2007)'nin hiyerarşik diklik koşulunu tanıtır ancak *"hesaplaması çok ağır olduğu için pratikte hazır yazılımı yoktur ve ALE tercih edilir"* der. İncelediğimiz **TreeHFD (Bénard, 2025)** makalesi, kitaptaki bu boşluğu doğrudan dolduran ilk pratik ağaç algoritmasıdır.
- **Bölüm 22.4.3 (Accumulated Local Effects — ALE) ve Bölüm 22.4.4 (Tree Ensembles):** Molnar, ALE'nin varyans ayrıştırması sunmadığını, PDP'nin ise korelasyonda bozulduğunu vurgular. Hücre 34'teki deneyimiz hem PDP'nin korelasyonda bozulduğunu hem de TreeHFD ile ALE'nin etkileşimsiz $\sin(x_2)$ bileşenini koruduğunu doğrulamıştır.

---

## 8. SAVUNMA HAZIRLIĞI (Canlı Sunum Rehberi)

### 8.1. 5 Dakikalık Canlı Akış Planı
1. **Dakika 0:00 – 0:45 (Giriş & Problem — Hücre 1 & 6):**
   - *Ne Gösterilecek:* California Housing korelasyon ısı haritası (`Latitude` vs `Longitude`, $r = -0.9247$).
   - *Ne Söylenecek:* "Hocam, Bölüm 22'de klasik fANOVA özelliklerin bağımsız olduğunu varsayar. Oysa gerçek veride enlem ve boylam $-0.92$ korelasyonludur. Hooker (2007) bağımlı veriler için hiyerarşik dikliği tanımlamış ama integral hesabı çözülememiştir. Seçtiğim NeurIPS 2025 makalesi (TreeHFD), bu problemi ağaçların Kartezyen bölmeleri üzerinde en küçük kareler çözerek saniyeler içinde hesaplanabilir kılıyor."
2. **Dakika 0:45 – 2:00 (Ana Yeniden Üretim & TreeSHAP Kıyası — Hücre 12, 13, 16):**
   - *Ne Gösterilecek:* Rekonstrüksiyon tablosu (`max_depth=2` için %100.00, `max_depth=5` için %97.99) ve `TreeHFD` (Mavi) vs `TreeSHAP` (Kırmızı) Boylam (`Longitude`) ana etki grafiği.
   - *Ne Söylenecek:* "Makalenin ana grafiğini bire bir yeniden ürettik. Mavi eğri (TreeHFD) $-122.4^\circ$ boylamındaki San Francisco Körfezi fiyat zirvesini $+0.785$ sıçramayla tek bir temiz 1B fonksiyon olarak yakalarken, kırmızı noktalar (TreeSHAP) aynı boylamda diğer özelliklerin etkileşimi sızdığı için bulutlaşıyor ve tepeyi $+0.380$'e basıklaştırıyor."
3. **Dakika 2:00 – 3:00 (Teorik Garantiler & Kendi Verimiz — Hücre 19, 22, 23):**
   - *Ne Gösterilecek:* Yeşil/Sarı özellik doğrulama tablosu ve **Bike Sharing Demand** (`veri.csv`) `hour` ve `hour × workingday` grafikleri.
   - *Ne Söylenecek:* "Sıfır ortalama ($10^{-6}$) ve hiyerarşik diklik ($3.58 \times 10^{-3}$) sayısal olarak doğrulandı. Kendi verimiz olan Bike Sharing'de de sıcaklık ile hissedilen sıcaklık $r=0.9877$ korelasyonlu olmasına rağmen TreeHFD model varyansının %97.41'ini açıkladı ve iş günü sabah 08:00 / akşam 17:00 piklerini hafta sonu öğlen pikinden dik olarak ayırdı."
4. **Dakika 3:00 – 4:15 (Sınırlar & Korelasyon Simülasyonu — Hücre 27, 30, 34, 35):**
   - *Ne Gösterilecek:* Desteklenmeyen durumlar tablosu (`ValueError` testleri) ve $\rho \in \{0, 0.5, 0.9\}$ simülasyon grafiği.
   - *Ne Söylenecek:* "Sınır testlerimizde paketin yalnız XGBoost ve `numpy.ndarray` kabul ettiğini, kategorik girdide hata verdiğini ve C++ TreeSHAP'tan ~5-10 kat yavaş olduğunu gösterdik. Ancak $\rho=0.9$ korelasyonlu simülasyonda TreeHFD, analitik Hoeffding bileşenlerini TreeSHAP'tan 4 kat daha düşük hatayla yakaladı."
5. **Dakika 4:15 – 5:00 (Özet Tablo & Sonuç — Hücre 37, 38):**
   - *Ne Gösterilecek:* "Ne Yeniden Üretildi?" tablosu ve 7 maddelik kazanç/kayıp özeti.

### 8.2. Beklenen 12 Jüri Sorusu ve Kısa Doğru Cevaplar
1. **Soru:** *Klasik fANOVA ile TreeHFD arasındaki temel fark nedir?*  
   **Cevap:** Klasik fANOVA özelliklerin bağımsız ($\mathbb{P}(X_j, X_k) = \mathbb{P}(X_j)\mathbb{P}(X_k)$) olduğunu varsayar; TreeHFD ise özellikler bağımlı olduğunda ampirik ortak dağılım üzerinde hiyerarşik diklik ($\mathbb{E}[f_j(X_j)f_{jk}(X_j, X_k)] = 0$) kısıtını ağaç bölmelerinde en küçük karelerle çözer.
2. **Soru:** *Hiyerarşik diklik (hierarchical orthogonality) tam olarak ne demektir?*  
   **Cevap:** Bir etkileşim bileşeni $f_S(X_S)$'nin, kendi alt kümesi olan hiçbir alt bileşenle ($U \subset S$, örn. ana etkiler $f_j(X_j)$) kovaryansının olmaması ($\mathbb{E}[f_U(X_U)f_S(X_S)] = 0$), yani etkileşim teriminin ana etkilere bilgi sızdırmaması demektir.
3. **Soru:** *Bağımlı girdilerde neden tüm bileşenler birbirine dik olamaz (tam diklik neden imkânsızdır)?*  
   **Cevap:** $X_1$ ve $X_2$ birbiriyle korelasyonluysa, onların sıfırdan farklı tek değişkenli fonksiyonları $f_1(X_1)$ ve $f_2(X_2)$ genel olarak $\mathbb{E}[f_1(X_1)f_2(X_2)] \neq 0$ kovaryansına sahip olur. Bu yüzden yalnızca alt-küme ile üst-küme ($f_1$ ile $f_{12}$) arasında diklik (hiyerarşik diklik) zorlanabilir.
4. **Soru:** *TreeHFD'de `max_depth=2` iken rekonstrüksiyon hatası $10^{-6}$ iken `max_depth=5` olduğunda neden %2.01'lik bir artık varyans kalıyor?*  
   **Cevap:** `max_depth=2` olan bir ağaçta bir kök-yaprak yolunda en fazla 2 özellik bölünebilir, yani ağaçta en fazla 2. mertebe etkileşim vardır. `max_depth=5` olduğunda ise ağaçlarda 3., 4. ve 5. mertebe etkileşimler de bulunur; `interaction_order=2` seçildiğinde bu yüksek mertebeli etkileşimler artık terimde ($y_{\text{xgb}} - y_{\text{hfd}}$) kalır.
5. **Soru:** *TreeSHAP'ın boylam (`Longitude`) ana etkisi grafiğinde neden tek bir eğri yerine kırmızı bir nokta bulutu görüyoruz?*  
   **Cevap:** Çünkü TreeSHAP'ın köşegen terimi $\Phi_{j,j}(\mathbf{x}^{(i)})$, yalnızca $x_{i,j}$'nin bir fonksiyonu değildir; ağaçlardaki 3+ mertebe etkileşimleri ve bağımlı yolları içerdiği için aynı boylamdaki farklı gözlemlerde diğer özelliklerin değerlerine göre farklı değerler alır.
6. **Soru:** *`treehfd_model.predict(X)` çıktısı kesişim sabitini ($\eta_0$) içeriyor mu?*  
   **Cevap:** Hayır. Kaynak kodundan doğruladığımız üzere `predict(X)` yalnızca sıfır ortalamalı `(y_main, y_order2)` matrislerini döndürür. Kesişim sabiti (`base_score` + ağaç ortalamaları) `treehfd_model.eta0` içinde saklanır ve tam tahmin $\eta_0 + \sum y_{\text{main}} + \sum y_{\text{order2}}$ ile bulunur.
7. **Soru:** *Simülasyonda $f(x_1, x_2, x_3) = x_1 + \sin(x_2) + x_1 x_3$ fonksiyonunda korelasyon $\rho=0.9$ olduğunda TreeHFD'nin $f_1(x_1)$ ana etkisi neden doğrusal $x_1$'den sapıyor?*  
   **Cevap:** Çünkü $(X_1, X_3)$ korelasyonlu olduğunda $\mathbb{E}[X_1 X_3 \mid X_1] = \rho X_1^2$'dir. Hoeffding diklik tanımı gereği çarpım teriminin $X_1$ ile açıklanabilen kısmı ana etkiye aktarılır ve $X_1$ üzerinde $\frac{\rho}{1+\rho^2}(X_1^2 - 1)$ şeklinde analitik bir karesel terim indüklenir.
8. **Soru:** *ALE ile TreeHFD arasındaki en kritik fark nedir?*  
   **Cevap:** ALE koşullu kısmi türevleri biriktirir ve varyans ayrıştırması yapmaz; TreeHFD ise $L^2$ izdüşümüne (koşullu beklentiye) dayanır, her ağacı kapalı formda ayrıştırır ve bileşenlerin model tahminini toplamsal olarak yeniden kurmasını sağlar.
9. **Soru:** *`treehfd` paketi kategorik değişkenleri veya eksik verileri (`NaN`) nasıl ele alıyor?*  
   **Cevap:** Yerel kategorik değişkenleri (`enable_categorical=True`) hiç desteklemez ve `ValueError` verir; önceden one-hot kodlama şarttır. Eksik değerlerde (`NaN`) ise `UserWarning` verir ve sadakat bozulur.
10. **Soru:** *Derin ağaçlarda (`max_depth=7+`) TreeHFD neden yavaşlar ve nasıl hızlandırılır?*  
    **Cevap:** Kartezyen bölümleme hücre sayısı üstel büyüdüğü için en küçük kareler matrisi büyür. `fit(X, depth_variable=(5, 4))` parametresiyle ana etki ve etkileşim değişken seçimi ağacın üst seviyeleriyle sınırlandırılarak hız korunur.
11. **Soru:** *Kendi veriniz olan Bike Sharing'de `hour × workingday` etkileşimi bize iş problemi açısından ne söylüyor?*  
    **Cevap:** Günün saatinin bisiklet talebine etkisi sabit değildir; iş günlerinde (`workingday=1`) sabah 08:00 ve akşam 17:00-18:00'de pozitif etkileşim sıçraması (işe gidiş-dönüş), hafta sonlarında (`workingday=0`) ise öğle saatlerinde pozitif etkileşim sıçraması vardır.
12. **Soru:** *TreeHFD'yi bir Sinir Ağı veya LightGBM modelini açıklamak için kullanabilir miyiz?*  
    **Cevap:** Doğrudan kullanamayız (`ValueError` verir); ancak kara kutu modelin tahminleri üzerine yüksek sadakatli ($R^2 > 0.98$) bir vekil (*surrogate*) XGBoost modeli eğitip ona TreeHFD uygulayarak dolaylı olarak ayrıştırabiliriz.

### 8.3. Emin Olmadığım / Açık Kalan Noktalar
- Makalede teorik asimptotik yakınsama (*consistency*) kanıtları yoğunluk fonksiyonunun alt-sınırlı (*bounded away from zero*) olması varsayımına dayanmaktadır; aşırı eşdoğrusal ($r = 0.9877$, `temp` vs `feel_temp`) durumda Kartezyen hücrelerin çoğu boş kaldığında en küçük kareler çözümünün hangi regülarizasyon/sözde-ters (*pinv*) toleransını kullandığı pratik açıdan incelenmeye açıktır.

### 8.4. Canlıda Hata Çıkarsa B Planı
1. **Ağ / Veri İndirme Hatası:** Hem `veri_california.csv` hem de `veri.csv` klasörde yerel olarak mevcuttur; notebook önce yerel CSV dosyalarını okur, internete hiç ihtiyaç duymaz.
2. **Süre / Bellek Sıkışması:** `FAST = True` bayrağı sayesinde hesaplanmış sonuçlar sistem geçici dizininden (`tempfile.gettempdir()`) milisaniyeler içinde yüklenir. Geçici dizin silinmiş olsa bile tüm hücreler `depth_variable` optimizasyonu sayesinde tek tek $\le 25$ saniyede, toplamda $\sim 90$ saniyede çalışmaktadır.
3. **Çekirdek Çökmesi:** `analiz.ipynb` zaten tüm çıktıları ve gömülü grafikleri (`dpi=100`) içinde barındıracak şekilde çalıştırılıp kaydedilmiştir; doğrudan mevcut çıktılar üzerinden sunuma kesintisiz devam edilebilir.

---

## 9. Kaynakça
1. **Bénard, C. (2025).** *Tree Ensemble Explainability through the Hoeffding Functional Decomposition and TreeHFD Algorithm*. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025). https://openreview.net/forum?id=dRLWcpBQxS
2. **Molnar, C. (2022).** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (Chapter 22: Functional Decomposition). https://christophm.github.io/interpretable-ml-book/functional-decomposition.html
3. **Hoeffding, W. (1948).** *A Class of Statistics with Asymptotically Normal Distribution*. The Annals of Mathematical Statistics, 19(3), 293–325.
4. **Hooker, G. (2007).** *Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables*. Journal of Computational and Graphical Statistics, 16(3), 709–732.
5. **Apley, D. W., & Zhu, J. (2020).** *Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models*. Journal of the Royal Statistical Society Series B, 82(4), 1059–1086.
6. **Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S. I. (2020).** *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence, 2(1), 56–67.
