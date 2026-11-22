# Veriyi Temsil Eden Örnekler: Ağırlıklı Prototipler (ProtoDash)

**Gurumoorthy ve ark. (2019), *Efficient Data Representation by Selecting Prototypes with Importance Weights* üzerine ayrıntılı Türkçe anlatım**

Makale: K. S. Gurumoorthy, A. Dhurandhar, G. Cecchi, C. Aggarwal. IEEE ICDM 2019. arXiv:1707.01212 · Kod: AIX360 (`ProtoDashExplainer`)
Temel yöntem: B. Kim, R. Khanna, O. Koyejo. *Examples are not Enough, Learn to Criticize! Criticism for Interpretability.* NeurIPS 2016 (MMD-critic)
Konu: Molnar, *Interpretable Machine Learning*, Bölüm 26 (Prototypes and Criticisms) · Makale tipi: Seçenek 3 (Geliştirme / Modifikasyon)
Açıklanabilir Yapay Zeka dersi, makale yeniden üretimi

> Bu belge makalenin çevirisi değildir. Makalenin akışını izleyerek fikirlerini, yöntemini ve deneylerini kendi anlatımımla ve ek açıklamalarla aktarır. Yeniden üretim bulguları (Bölüm 8) ve terim sözlüğü (Bölüm 9) bana aittir. Kesin ifadeler, teoremler ve şekiller için orijinal makaleye başvurulmalıdır.


**İçindekiler**

1. Problem: Bir veri kümesini örneklerle nasıl anlatırız?
2. Arka plan: Prototipler, eleştiriler ve MMD-critic
3. MMD-critic'in eksikleri ve ProtoDash'in fikri
4. ProtoDash algoritması
5. Deney düzeni
6. Sonuçlar ve yorumları
7. Makalenin konumu, sınırları ve sonrası
8. Kendi yeniden üretimimde gözlenenler
9. Terim sözlüğü

## 1. Problem: Bir veri kümesini örneklerle nasıl anlatırız?

Binlerce ya da milyonlarca satırlık bir veri kümesini bir insan tek tek inceleyemez. Ona "bu veri neye benziyor?" sorusunu cevaplamanın pratik bir yolu, birkaç **gerçek örnek** göstermektir. Bu örnekler iki iş görür:

- **Veriyi anlamak:** Hangi tür müşteriler, hangi tür görüntüler baskın?
- **Modeli anlamak:** Bir modelin hangi örneklerde iyi, hangilerinde kötü çalıştığını bu temsilcilere bakarak görmek.

Burada iki tür örnek vardır. **Prototip**, verinin tamamını iyi temsil eden bir örnektir (tipik olan). **Eleştiri (criticism)** ise prototiplerin iyi temsil etmediği, tipik olmayan ama veride gerçekten bulunan bir örnektir. Yalnızca prototipler göstermek yanıltıcıdır, çünkü veri kümesi tek tip görünür. Eleştiriler "yine de bu da var" der.

Makalenin motivasyonu şudur: Prototipleri seçmek için bir ölçüt gerekir. Ölçüt, seçilen prototip kümesinin veri dağılımını ne kadar iyi **yaklaştırdığıdır**. Üstelik her prototipin veriyi temsil etme payı eşit olmayabilir. Bir prototip büyük bir kümeyi, bir diğeri küçük bir kümeyi temsil edebilir. Bu bilgiyi ağırlık olarak vermek hem daha doğru bir temsil hem de daha yorumlanabilir bir sonuç sağlar.

> **Tek cümlelik özet:** ProtoDash, MMD-critic'in prototip seçimini her prototipe bir önem ağırlığı atayacak, hedef kümeyi aday havuzundan ayırabilecek ve çok daha hızlı çalışacak biçimde genelleştirir.

## 2. Arka plan: Prototipler, eleştiriler ve MMD-critic

### 2.1 Maksimum ortalama sapma (MMD)

İki dağılımın ne kadar farklı olduğunu ölçmenin bir yolu, bir **çekirdek (kernel)** *k* altında ortalamalarını karşılaştırmaktır. *X* veri kümesi (*n* örnek) ve seçilen *m* prototip *P* için:

$$
\mathrm{MMD}^2(X, P) = \frac{1}{n^2}\sum_{i,j} k(x_i, x_j) \;-\; \frac{2}{nm}\sum_{i,j} k(x_i, p_j) \;+\; \frac{1}{m^2}\sum_{i,j} k(p_i, p_j)
$$

Yaygın seçim RBF çekirdeğidir: $k(x, x') = \exp\!\left(-\gamma\,\lVert x - x'\rVert^2\right)$. Çekirdek, iki noktanın birbirine ne kadar benzediğini 0-1 arasında verir. MMD² küçüldükçe *P*'nin dağılımı *X*'inkine yaklaşır.

### 2.2 Prototip seçimi

MMD-critic, MMD²'yi en çok azaltan *m* örneği **açgözlü** biçimde seçer: Her adımda, eklendiğinde MMD²'yi en çok düşüren aday kümeye girer. Her prototip eşit ağırlıkla sayılır (1/*m*).

### 2.3 Eleştiriler ve tanık fonksiyonu

Prototipler seçildikten sonra, veriyle prototipler arasındaki yoğunluk farkını gösteren **tanık fonksiyonu** (witness function) hesaplanır:

$$
\mathrm{witness}(x) = \frac{1}{n}\sum_{i=1}^{n} k(x, x_i) \;-\; \frac{1}{m}\sum_{j=1}^{m} k(x, p_j)
$$

Bu değer pozitifse, *x*'in bulunduğu bölgede veri yoğun ama prototipler zayıf: prototipler bu bölgeyi **yetersiz temsil eder**. Negatifse prototipler orayı **fazla** temsil eder. |witness|'ı büyük olan noktalar eleştiri adayıdır. MMD-critic, eleştirileri bu değeri büyük olan noktalar arasından seçer (Kim ve ark. seçimi çeşitli kılmak için ek bir düzenleme de uygular; burada ayrıntısına girmiyorum).

> **Benzetme:** Bir şehri anlatmak için gösterilecek mahalleleri seçiyorsunuz. Prototipler, şehrin büyük kısmını yansıtan mahallelerdir. Eleştiriler ise "bunlar şehrin geri kalanını yansıtmıyor, ama şu tuhaf bölge de gerçekten var" dediğiniz yerlerdir.

## 3. MMD-critic'in eksikleri ve ProtoDash'in fikri

MMD-critic işe yarar ama makalenin işaret ettiği üç sınırı vardır:

1. **Eşit ağırlık.** Her prototip aynı payla sayılır. Büyük bir kümeyi temsil eden prototip ile küçük bir kümeyi temsil eden prototip birbirinden ayırt edilemez. Kullanıcı hangisinin "ne kadar önemli" olduğunu görmez.
2. **Hedef ve havuz aynı küme.** Prototipler, temsil edilecek kümenin içinden seçilir. Oysa pratikte çoğu zaman **başka bir kümeyi** temsil etmek isteriz: Örneğin tüm müşterilerden (havuz) yalnızca riskli müşteri grubunu (hedef) temsil eden örnekler bulmak.
3. **Hız.** Daha önceki ağırlıklı yaklaşımlar (makale bunlardan ProtoGreedy'yi de sayar) her adımda tüm adaylar için bir optimizasyon problemi çözmek zorundadır. Aday sayısı büyüdükçe bu pahalıdır.

**ProtoDash** bu üç sorunu birlikte çözer:

- Her prototipe bir **negatif olmayan ağırlık** atar.
- **Hedef küme** ile **aday havuzunu** ayırır: Hedef küme *X*<sup>(1)</sup>, aday havuzu *X*<sup>(2)</sup>.
- Aday seçimini **gradyan** üzerinden yaparak çok daha hızlı çalışır.

## 4. ProtoDash algoritması

### 4.1 Amaç fonksiyonu

Hedef küme *X*<sup>(1)</sup> (*n* örnek), aday havuzu *X*<sup>(2)</sup> (*N* aday) olsun. Havuzdaki adayların çekirdek matrisi *K*<sub>ij</sub> = *k*(*p*<sub>i</sub>, *p*<sub>j</sub>) ve her adayın hedef kümeye ortalama benzerliği

$$
\mu_j = \frac{1}{n}\sum_{x \in X^{(1)}} k(p_j, x)
$$

olsun. Seçilen prototip indeksleri *S* ve ağırlıkları *w* ≥ 0 için ağırlıklı MMD²'yi küçültmek, sabit terimler hariç, şu fonksiyonu büyütmeye denktir:

$$
\ell(w) = w^{\top}\mu_S \;-\; \tfrac{1}{2}\,w^{\top} K_{SS}\, w, \qquad w \ge 0
$$

Birinci terim, prototiplerin hedef kümeye benzerliğini ödüllendirir. İkinci terim, prototiplerin birbirine benzemesini cezalandırır (aynı yeri iki kez temsil etme). Her *w*<sub>i</sub>, o prototipin önem payıdır. Ağırlıkları toplamı 1 olacak şekilde normalize edince "bu prototip hedefin yüzde kaçını temsil ediyor" okumasını verir.

### 4.2 Adım adım

Prototip sayısı *m* verilsin. Başlangıçta *S* boştur.

1. **Aday seçimi (gradyan):** Mevcut ağırlıklar altında her aday *j* için gradyan hesaplanır: $\nabla_j = \mu_j - (K w)_j$. Gradyanı en büyük olan, henüz seçilmemiş aday *S*'ye eklenir. Gradyan, "bu aday eklenirse amaç ne kadar artar?" sorusunun doğrusal bir tahminidir.
2. **Ağırlık çözümü:** Yeni *S* için ℓ(*w*) en büyüklenir, *w* ≥ 0 kısıtıyla. Bu, küçük bir **negatif olmayan en küçük kareler** (NNLS) problemidir. Uygulamada *K*<sub>SS</sub>'nin Cholesky ayrıştırmasıyla çözülür.
3. *m* adım tamamlanana kadar tekrar edilir.

Ağırlığı 0 çıkan prototip "seçildi ama artık katkı vermiyor" demektir. Bu, ağırlıkların önem sıralaması vermesinin pratik yansımasıdır.

### 4.3 ProtoGreedy ile fark

**ProtoGreedy**, her adımda **tüm adaylar** için ağırlık problemini çözüp amaç fonksiyonunu hesaplar ve gerçekten en çok artıran adayı seçer. Bu daha titizdir ama aday sayısı kadar optimizasyon çözmek demektir. ProtoDash aday seçmek için yalnızca gradyana bakar, yani tek bir hızlı hesapla adayları sıralar ve ağırlık problemini adım başına yalnızca bir kez çözer. Makale, iki yöntemin seçim kalitesinin yakın olduğunu ve ProtoDash'in çok daha hızlı olduğunu raporlar. Makale yaklaşıklık garantilerinden de söz eder; bunları bu çalışmada doğrulamadım, ayrıntılar için makaleye bakılmalıdır.

### 4.4 Ağırlıkların yorumu

- **Yüksek ağırlıklı** prototip: Hedef kümenin büyük bir kısmının yakınında, onu temsil ediyor.
- **Düşük ağırlıklı** prototip: Daha az sayıda örneği ya da hedefin kenarını temsil ediyor. Makale, düşük ağırlıklı prototiplerin **eleştiri gibi** davranabileceğini söyler. Bu bir iddiadır; deneyle sınanması Bölüm 8.4'tedir.

## 5. Deney düzeni

Makalenin ana deneyi **skewli MNIST** üzerindedir.

### 5.1 Kurgu

- Hedef rakam *d* seçilir.
- **Hedef küme** *X*<sup>(1)</sup>: Örneklerin *s*% kadarı rakam *d*, geri kalanı diğer rakamlardan. *s* 30'dan %100'e kadar artırılır (skew).
- **Aday havuzu** *X*<sup>(2)</sup>: Tüm rakamlardan, skewsiz.
- Her yöntem havuzdan *m* prototip seçer.
- Değerlendirme: Taze bir skewli örneklemde, seçilen prototiplerle **1-en yakın komşu (1-NN)** sınıflandırması yapılır ve doğruluk ölçülür.

Skew arttıkça hedef küme havuzdan uzaklaşır. Ağırlıklı ve hedefe odaklı seçimin burada avantajlı olması beklenir.

### 5.2 Karşılaştırılan yöntemler

ProtoDash, ProtoGreedy, MMD-critic (eşit ağırlıklı açgözlü seçim), K-Medoids ve rastgele seçim.

### 5.3 Ölçütler

- **1-NN doğruluğu** (yukarıdaki kurgu),
- **MMD²** (hedef kümeyle seçilen prototipler arasındaki fark, küçük iyi),
- **Süre** (seçimin ne kadar sürdüğü),
- **Ağırlık dağılımı** (%100 skewde ağırlığın hedef sınıfa gidip gitmediği).

## 6. Sonuçlar ve yorumları

Makalenin raporladığı nitel bulgular (özet):

1. **Skew arttıkça ProtoDash'in kazancı büyür.** Düşük skewde basit yöntemler (K-Medoids) rekabetçidir; hedef havuzdan uzaklaştıkça ağırlıklı seçim öne geçer.
2. **ProtoDash ≈ ProtoGreedy doğrulukta**, ama ProtoDash çok daha hızlıdır.
3. **%100 skewde ağırlık hedef sınıfta toplanır.** Diğer sınıflardan seçilen prototipler küçük ağırlık alır.
4. **Düşük ağırlıklı prototipler eleştiri gibi davranır.** Makale bunu yorum olarak sunar, ayrı bir sayısal deneyle ölçmez.

**Yorum.** Bu dört bulgu birlikte şunu söyler: Ağırlık bir yan ürün değildir. Hedefe odaklanmayı mümkün kılan ve sonucu yorumlanabilir yapan asıl bileşendir. Hız ise yöntemin büyük veriye uygulanabilir olmasını sağlar.

## 7. Makalenin konumu, sınırları ve sonrası

### 7.1 Literatürdeki yeri

- **MMD-critic'in devamı.** Kim ve ark. (2016) prototip ve eleştiri fikrini kurdu. ProtoDash bu çerçevenin ağırlık ve hedef/havuz ayrımıyla genişletilmiş biçimidir.
- **Açıklanabilirlik aracı.** ProtoDash, IBM'in AIX360 kütüphanesinde `ProtoDashExplainer` olarak bulunur ve "bir kararı, benzer eğitim örnekleriyle açıkla" tarzı örnek tabanlı açıklamalarda kullanılır.
- **Benzer fikirler.** Örnek tabanlı açıklamalar (k-NN tarzı), etki fonksiyonları ve veri özetleme yöntemleri aynı soruya farklı yönlerden yaklaşır.

### 7.2 Eleştirel değerlendirme (kendi yorumum)

- **Çekirdek ve γ seçimi.** Bütün sonuç RBF çekirdeğine ve genişlik parametresine bağlıdır. Seçilen prototiplerin kimliği bu seçime duyarlıdır (Bölüm 8.3).
- **Karışık tip veri.** Çekirdek, Öklid uzayında benzerlik varsayar. Kategorik değişkenler one-hot ile gömüldüğünde "benzerlik" tartışmalı hâle gelir.
- **Görüntü ve tablo farkı.** MNIST'te prototip bir resimdir; bir bakışta anlaşılır. Tabloda prototip bir satırdır ve okunması daha emek ister.
- **Doğruluk ölçütü.** 1-NN doğruluğu temsil kalitesinin dolaylı bir ölçüsüdür. Asıl amaç dağılımı temsil etmektir (MMD²); iki ölçüt her zaman aynı yöne bakmaz.
- **Eleştirilerin seçimi.** ProtoDash ağırlık verir ama eleştiri seçimi için ayrı bir kural gerekir. Burada tanık fonksiyonunu kullanıyorum.

## 8. Kendi yeniden üretimimde gözlenenler

ProtoDash'i, ProtoGreedy'yi ve MMD-critic'in eşit ağırlıklı açgözlü seçimini kendi kodumla yeniden uyguladım (`analiz.ipynb`; veri: `veri.csv`). Farklarım: AIX360 kullanmadım; ProtoDash'te gradyanla aday seçip NNLS ile ağırlık çözdüm; K-Medoids yerine **KMeans + en yakın havuz noktası** yaklaşığı kullandım (gerçek PAM değil); deney boyutlarını (havuz, örneklem, tekrar sayısı) kendim seçtim. Sayılar makaledekiyle birebir aynı çıkmaz, **eğilimler** karşılaştırılır. Aşağıdaki "tutuyor mu" tablolarındaki eşikler bana aittir.

### 8.1 Orijinal veri (skewli MNIST)

70.000 görüntüden rastgele 10.000'i alındı (784 özellik). Havuz 6.000, hedef küme örneklemi (A) 2.000, değerlendirme (B) 2.000 görüntü. Her skew düzeyi için 8 tekrar, her tekrarda rastgele bir hedef rakam, *m* = 10 prototip, aday havuzu 500, hedef örneklemi 200. γ = 0,0048 (medyan sezgisi).

**Ortalama 1-NN doğruluğu**

| Yöntem | %30 | %50 | %70 | %90 | %100 |
|---|---|---|---|---|---|
| ProtoDash | 0,378 | 0,465 | 0,589 | 0,816 | 0,876 |
| ProtoGreedy | 0,394 | 0,514 | 0,613 | 0,841 | 0,895 |
| MMD-critic (eşit ağırlık) | 0,399 | 0,491 | 0,611 | 0,794 | 0,896 |
| K-Medoids~ | 0,515 | 0,572 | 0,666 | 0,846 | 0,931 |
| Rastgele | 0,326 | 0,386 | 0,372 | 0,384 | 0,369 |

**Ortalama MMD²** (küçük iyi)

| Yöntem | %30 | %50 | %70 | %90 | %100 |
|---|---|---|---|---|---|
| ProtoDash | 0,0154 | 0,0155 | 0,0163 | 0,0138 | 0,0163 |
| ProtoGreedy | 0,0148 | 0,0149 | 0,0150 | 0,0134 | 0,0152 |
| MMD-critic (eşit ağırlık) | 0,0164 | 0,0159 | 0,0162 | 0,0145 | 0,0172 |
| K-Medoids~ | 0,0233 | 0,0244 | 0,0260 | 0,0274 | 0,0268 |
| Rastgele | 0,0449 | 0,0490 | 0,0611 | 0,1001 | 0,1050 |

**Makale iddiası ↔ bu yeniden üretim** (eşikler bana ait):

| Makale iddiası | Ölçüt (benim eşiğim) | Bu yeniden üretimde | Tutuyor mu |
|---|---|---|---|
| (a) Skew arttıkça ProtoDash'in K-Medoids'e göre kazancı büyür | kazanç(%100) > kazanç(%30) | −0,138 → −0,054 | Evet (ama kazanç hep negatif) |
| (b1) ProtoDash ≈ ProtoGreedy doğrulukta | ortalama \|fark\| < 0,02 | 0,027 | Hayır |
| (b2) ProtoDash ProtoGreedy'den çok daha hızlı | hız oranı > 5× | 186,4× | Evet |
| (c) %100 skewde ağırlık hedef sınıfta toplanır | pay > 0,80 | 0,742 | Hayır |

**Yorum (4 iddianın 2'si tuttu).**
- **Hız (b2) güçlü biçimde yeniden üretildi.** ProtoDash, ProtoGreedy'den yaklaşık 186 kat hızlı.
- **Ağırlıklı yöntemler temsilde (MMD²) K-Medoids~'in ve rastgelenin açık farkla önünde.** ProtoDash 0,014-0,016, K-Medoids~ 0,023-0,027, rastgele 0,045-0,105.
- **Ancak 1-NN doğruluğunda ProtoDash hiçbir skew düzeyinde K-Medoids~'i geçemedi.** (a) eşiği karşılandı çünkü fark −0,138'den −0,054'e kapandı, ama bu "ProtoDash kazanıyor" demek değil; "ProtoDash geride kalıyor ama fark azalıyor" demek. K-Medoids~'in doğrudan sınıflandırma için temsilci seçen bir yaklaşık olması bunu açıklayabilir: MMD² ile doğruluk aynı şeyi ölçmüyor.
- **(b1) tutmadı.** ProtoDash ile ProtoGreedy arasındaki ortalama doğruluk farkı 0,027, eşiğim 0,02'nin biraz üstünde. ProtoGreedy'nin MMD²'si de her skewde biraz daha iyi, yani ikisi yakın ama ProtoGreedy hâlâ biraz önde.
- **(c) tutmadı.** %100 skewde ProtoDash ağırlığının %74'ü hedef sınıfa gitti (eşiğim %80). Rastgele seçimin payı yaklaşık %14, yani ağırlık hedefe belirgin biçimde yöneliyor, ama makalenin söylediği gibi neredeyse tamamen toplanmıyor. K-Medoids~ sayıyla %92,5 pay aldı.
- **Neden fark var?** Deney boyutlarını (havuz 500, örneklem 200, 8 tekrar) kendim seçtim, K-Medoids gerçek PAM değil ve eşikler benim. Bu yüzden sonuç yöntemin çürütüldüğü değil, **bu kurguda iddiaların yalnızca bir kısmının görüldüğü** şeklinde okunmalı.

### 8.2 Kendi verim (Kaggle German Credit, 1000 başvuru)

Hedef sınıf **riskli müşteri** (Risk = bad, %30). Özellikler: Age, Job, log Credit amount, Duration, Sex standartlaştırılmış; Housing, Saving accounts, Checking account, Purpose one-hot. RBF çekirdeği, γ medyan sezgisiyle seçildi. Havuz: tüm müşteriler (%60, 600 başvuru). Hedef küme: ayrı bir %20'lik kısımdaki `bad` müşteriler.

**Okunabilir prototipler (ağırlıkça sıralı, 8 prototip).** Ağırlık "bu müşteri hedef grubun yüzde kaçını temsil ediyor" demektir:

| Ağırlık % | Yaş | Cinsiyet | Konut | Tasarruf | Vadesiz hesap | Kredi tutarı | Vade (ay) | Amaç | Risk |
|---|---|---|---|---|---|---|---|---|---|
| 28,5 | 32 | erkek | own | little | little | 1938 | 24 | radio/TV | bad |
| 15,4 | 31 | kadın | own | little | no account | 1736 | 12 | furniture/equipment | good |
| 11,7 | 28 | kadın | rent | quite rich | moderate | 4113 | 24 | car | bad |
| 10,1 | 25 | erkek | own | unknown/no account | little | 8072 | 30 | business | good |
| 10,0 | 50 | erkek | free | little | moderate | 6224 | 48 | education | bad |
| 9,8 | 25 | erkek | rent | moderate | moderate | 1264 | 15 | car | bad |
| 9,2 | 30 | kadın | own | little | moderate | 5096 | 48 | furniture/equipment | bad |
| 5,2 | 66 | erkek | free | little | little | 1526 | 12 | car | good |

- **Ağırlık hedefe gidiyor.** Havuzda `bad` oranı %30 iken, prototiplerin toplam ağırlığının %69'u `bad` müşterilere düştü. En ağır prototip (%28,5) 32 yaşında, 24 ay vadeli, küçük tasarruflu bir `bad` müşteri: hedef grubun "tipik" üyesi.
- **Hedef bad olmasına rağmen good prototipler de seçildi.** Havuz tüm müşterileri içerdiği için ProtoDash, hedef kümeye çekirdek altında yakın olan `good` müşterileri de aldı (toplam ağırlık payı yaklaşık %31). Bu, "bad" ile "good" müşterilerin bu özellik uzayında tamamen ayrışmadığını gösterir ve tablo verisinde dürüst bir uyarıdır.
- **Eleştiriler (en büyük |tanık| değeri, 5 müşteri).** Hepsi `radio/TV` amaçlı, vadesi 9-30 ay arası, çoğu 22-35 yaşında erkek `bad` müşteriler. Beş tanık değerinin hepsi negatif (−0,032 ile −0,040). Negatif değer, prototiplerin bu bölgeye veri yoğunluğuna göre **fazla** ağırlık verdiği anlamına gelir; yani bu noktalar klasik anlamda "yetersiz temsil edilen" değil, "aşırı temsil edilen" bir bölgede (genç, kısa vadeli `radio/TV` kredileri). En büyük |tanık| değerleri negatif çıktığı için, pozitif (yetersiz temsil edilen) tarafta bundan güçlü bir sapma olmadığı anlaşılıyor. Bu, eleştirileri yalnızca mutlak değere göre seçmenin işaretini ayrıca raporlamayı gerektirdiğini gösterir.

**Skew deneyi (R = 15 tekrar, m = 10 prototip, havuz 300).** Ortalama MMD² (küçük iyi):

| Yöntem | %30 | %50 | %70 | %90 | %100 |
|---|---|---|---|---|---|
| ProtoDash | 0,0075 | 0,0078 | 0,0080 | 0,0086 | 0,0090 |
| ProtoGreedy | 0,0082 | 0,0094 | 0,0095 | 0,0100 | 0,0106 |
| MMD-critic (eşit ağırlık) | 0,0102 | 0,0105 | 0,0109 | 0,0114 | 0,0118 |
| K-Medoids~ | 0,0262 | 0,0227 | 0,0260 | 0,0257 | 0,0266 |
| Rastgele | 0,0491 | 0,0569 | 0,0594 | 0,0669 | 0,0633 |

1-NN doğruluğu aynı eğilimi gösterir ama gürültülüdür: %100 skewde ProtoDash 0,505, ProtoGreedy 0,419, MMD-critic 0,483, K-Medoids~ 0,477, rastgele 0,293. ProtoDash'in K-Medoids~'e doğruluk farkı %30 skewde −0,019, %100 skewde +0,028 oldu.

- **Temsil açısından ProtoDash her skew düzeyinde en iyisi.** MMD²'de ProtoDash eşit ağırlıklı MMD-critic'ten 5/5 skew düzeyinde daha iyi, K-Medoids~'ten ve rastgeleden çok daha iyi. Ağırlık, aynı prototip sayısıyla hedefi daha iyi yaklaştırıyor.
- **Eğilim makalenin yönünde.** Skew arttıkça ProtoDash'in K-Medoids~'e doğruluk kazancı büyüdü (−0,019 → +0,028). Ancak bu 15 tekrarlı, küçük veride bir **eğilim**dir; tek tek sayılar gürültülüdür ve fark küçük.
- **ProtoDash ≈ ProtoGreedy.** MMD²'de ProtoDash hatta biraz daha iyi çıktı. ProtoGreedy'nin kendi amaç fonksiyonunu daha iyi optimize etmesi beklenir; benim ölçtüğüm MMD² ise normalize edilmiş ağırlıklarla hesaplanıyor ve örnekler küçük, bu yüzden sıralama tersine dönebiliyor. Sonuç "ProtoDash daha iyi" değil, "ikisi pratikte yakın" diye okunmalı.

### 8.3 Duyarlılık

- **Çekirdek genişliği γ.** γ'yı medyan sezgisinin 0,25×, 0,5×, 1×, 2×, 4× katına getirdiğimde, 1×'te seçilen 10 prototip kümesiyle diğerlerinin Jaccard benzerliği 0,05 ile 0,33 arasında kaldı. Yani **prototiplerin kimliği γ'ya çok duyarlı**. ProtoDash'in MMD²'sinin rastgele seçime oranı ise her γ için küçük kaldı (0,06 ile 0,38 arası): Yöntem her ayarda rastgeleden iyi, ama *hangi* müşterilerin seçildiği ayara bağlı. Bu yüzden γ seçimi sonuç raporlanırken mutlaka belirtilmelidir.
- **Prototip sayısı.** *m* = 2'den 40'a çıkarken ProtoDash'in MMD²'si 0,0670'ten 0,0011'e düştü. *m* = 10'da ProtoDash 0,0080, MMD-critic 0,0118, rastgele 0,0645. ProtoDash her *m* değerinde MMD-critic'ten iyi; *m* arttıkça ProtoDash'in oransal avantajı büyüyor: ProtoDash/MMD-critic MMD² oranı *m* = 2'de 0,88, *m* = 40'ta 0,61.

### 8.4 Uzantı: "düşük ağırlıklı prototipler eleştiri gibi davranır" iddiası

Makale bu iddiayı sayısal olarak ölçmez. Ben bunu hedef kümedeki yerel yoğunluk $\mu(x) = \frac{1}{n}\sum k(x, x')$ üzerinden sınadım: Düşük ağırlıklı prototipler gerçekten seyrek bölgede mi? 20 farklı havuz altörneğinde (*m* = 15, hedef %100 `bad`):

| Grup | Ortalama μ (hedef yoğunluğu) | Std |
|---|---|---|
| Yüksek ağırlıklı prototipler (üst 1/3) | 0,622 | 0,021 |
| Düşük ağırlıklı prototipler (alt 1/3) | 0,488 | 0,037 |
| MMD-critic eleştirileri | 0,631 | 0,023 |
| Rastgele havuz noktası | 0,576 | 0,008 |

- Ağırlık ile μ arasındaki Spearman korelasyonu ortalama +0,70; 20 altörneklemin 20'sinde pozitif.
- Düşük ağırlıklı prototipler hem yüksek ağırlıklılardan hem de **rastgele bir havuz noktasından** daha seyrek bölgede. İddia, yön olarak doğrulandı.
- **Ancak eleştiriler seyrek bölgede değil.** MMD-critic eleştirilerinin yoğunluğu (0,631) yüksek ağırlıklı prototiplerinkine (0,622) eşit ya da biraz yüksek. Bunun nedeni, tanık fonksiyonunun eleştirileri *hedef küme noktaları* arasından seçmesi ve bu noktaların kendi başına yoğun olabilmesidir. Yani "düşük ağırlık = eleştiri gibi **seyrek bölge**" doğru, "düşük ağırlıklı prototip = MMD-critic eleştirisi" ise **aynı şey değil**.
- **Uyarı:** Bu korelasyon kısmen yapısaldır. ProtoDash gradyanı büyük (yani μ büyük) adayı önce seçtiği için ilk seçilenler zaten yoğun bölgeden gelir. Bu yüzden asıl bulgu "rastgeleden de seyrek" kısmıdır.

## 9. Terim sözlüğü

| İngilizce | Türkçe | Kısa açıklama |
|---|---|---|
| Prototype | Prototip | Veriyi iyi temsil eden gerçek örnek |
| Criticism | Eleştiri | Prototiplerin iyi temsil etmediği gerçek örnek |
| MMD (Maximum Mean Discrepancy) | Maksimum ortalama sapma | İki dağılımın çekirdek altında ortalama farkı |
| Kernel | Çekirdek | İki örneğin benzerliğini veren fonksiyon (RBF gibi) |
| Bandwidth (γ) | Çekirdek genişliği | Benzerliğin uzaklıkla ne hızla düştüğünü belirler |
| Witness function | Tanık fonksiyonu | Veri ile prototipler arasındaki yoğunluk farkı |
| Importance weight | Önem ağırlığı | Prototipin hedefi temsil etme payı |
| Target set | Hedef küme | Temsil edilmek istenen veri |
| Candidate pool | Aday havuzu | Prototiplerin seçileceği veri |
| Skew | Çarpıklık (skew) | Hedef kümenin sınıf dağılımının havuzdan farklı olması |
| ProtoGreedy | ProtoGreedy | Her adımda tüm adayları değerlendiren yavaş ağırlıklı seçim |
| NNLS | Negatif olmayan en küçük kareler | w ≥ 0 kısıtlı en küçük kareler problemi |
| 1-NN | 1-en yakın komşu | En yakın prototipin etiketini alan sınıflandırıcı |
| Jaccard similarity | Jaccard benzerliği | İki kümenin kesişiminin birleşimine oranı |

---

Kaynak: K. S. Gurumoorthy, A. Dhurandhar, G. Cecchi, C. Aggarwal. "Efficient Data Representation by Selecting Prototypes with Importance Weights." IEEE ICDM 2019. arXiv:1707.01212 · Kod: AIX360 (github.com/Trusted-AI/AIX360)

Diğer anılan çalışmalar: Kim, Khanna, Koyejo 2016, "Examples are not Enough, Learn to Criticize!" (NeurIPS); Molnar, "Interpretable Machine Learning", Bölüm 26.
