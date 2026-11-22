# Açıklama Adil Görünüyor Diye Model Adil mi? Fairwashing

**Aïvodji ve ark. (2019), *Fairwashing: the risk of rationalization* üzerine ayrıntılı Türkçe anlatım**

Makale: U. Aïvodji, H. Arai, O. Fortineau, S. Gambs, S. Hara, A. Tapp. ICML 2019. arXiv:1901.09749 · Kod: github.com/aivodji/LaundryML
Konu: Molnar, *Interpretable Machine Learning*, Bölüm 25 (Global Surrogate Models) · Makale tipi: Seçenek 1 (Zafiyet / Manipülasyon)
Açıklanabilir Yapay Zeka dersi, makale yeniden üretimi

> Bu belge makalenin çevirisi değildir. Makalenin akışını izleyerek fikirlerini, yöntemini ve deneylerini kendi anlatımımla ve ek açıklamalarla aktarır. Yeniden üretim bulguları (Bölüm 9) ve terim sözlüğü (Bölüm 10) bana aittir. Kesin ifadeler, tablolar ve şekiller için orijinal makaleye başvurulmalıdır.


**İçindekiler**

1. Problem: Açıklama güvenilir bir denetim aracı mı?
2. Arka plan: Global surrogate nasıl çalışır?
3. Zafiyetin kaynağı: Sadakat, hatanın nerede olduğunu söylemez
4. Tehdit modeli ve LaundryML
5. Deney düzeni
6. Sonuçlar ve yorumları
7. Uzantı: Aklama tespit edilebilir mi? (FRAUD-Detect)
8. Makalenin konumu, sınırları ve sonrası
9. Kendi yeniden üretimimde gözlenenler
10. Terim sözlüğü

## 1. Problem: Açıklama güvenilir bir denetim aracı mı?

Kredi, işe alım ya da suç tekrarı riski gibi alanlarda kullanılan modellerin çoğu karmaşıktır ve çoğu zaman ticari olarak kapalıdır. Bir denetçi ya da düzenleyici kurum modelin içine bakamaz, yalnızca ona girdi verip çıktı alabilir. Bu durumda en yaygın yol, kara kutuyu **kendisinden daha basit ve okunabilir bir modelle taklit etmektir**: Kara kutunun kararlarını veri üzerinde toplar, bu kararları öğrenen küçük bir karar ağacı ya da kural listesi eğitir ve "kara kutu aslında şu kurallara göre karar veriyor" der. Bu basit modele **global surrogate** (vekil model) denir.

Denetçi bu açıklamaya şu iki sorudan geçerek güvenir:

1. **Sadakat (fidelity):** Vekil model, kara kutunun kararlarının ne kadarını doğru taklit ediyor? Yüksekse açıklama "doğru" kabul edilir.
2. **Adalet:** Vekil modelde cinsiyet, ırk gibi hassas bir özellik belirleyici mi? Değilse kara kutu da adil kabul edilir.

Bu iş akışının gizli bir varsayımı vardır: **Yüksek sadakatli bir açıklama, kara kutunun gerçek davranışını yansıtır.** Makalenin sorduğu soru bu varsayımın kırılgan olup olmadığıdır. Yazarların cevabı: Model sahibi kötü niyetliyse, kara kutunun adaletsizliğini gizleyen ama yine de yüksek sadakatli bir vekil model **bilerek üretebilir**. Yazarlar bu işleme **fairwashing** (adalet aklama) der. Bu, "açıklama adil görünüyor, o hâlde model adildir" çıkarımının geçersiz olabileceği anlamına gelir.

> **Tek cümlelik özet:** Sadakat tek bir ortalama sayıdır; vekil modelin hataları kara kutunun adaletsizliğini besleyen örneklere yığılabilir ve ortalama sadakat yine de yüksek kalabilir.

## 2. Arka plan: Global surrogate nasıl çalışır?

Molnar'ın anlattığı adımlar şöyledir:

1. Bir veri kümesi *X* seç (kara kutunun eğitim verisi olabilir, olmak zorunda değil).
2. Kara kutu *B* ile *X* için tahminleri al: *B*(*x*).
3. Yorumlanabilir bir model *I* seç (karar ağacı, doğrusal model, kural listesi).
4. *I*'yı gerçek etiketlerle **değil**, kara kutunun tahminleriyle eğit.
5. Vekilin kara kutuyu ne kadar iyi taklit ettiğini ölç.

İkili sınıflandırmada bu ölçü **sadakat**tir:

$$
\text{fid}(B, I) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\!\left[\,I(x_i) = B(x_i)\,\right]
$$

Regresyonda aynı iş için *R*² kullanılır. Burada önemli bir nokta vardır: Vekil model **veriyi** değil, **kara kutuyu** açıklar. Doğru etiketlerle ilgisi yoktur. Kara kutu yanlış kararlar veriyorsa, sadık bir vekil de aynı yanlışları öğrenir.

Vekil modelin avantajları: Her model türüyle çalışır, kurulumu basittir, tek bir küçük model tüm kara kutuyu özetler. Sınırları: Yorumlanabilir bir model karmaşık bir modeli her bölgede taklit edemez. Sadakat yüksek görünse bile vekil, kara kutunun yalnızca **bir bölümünü** yakalayabilir. Hangi bölümü yakaladığı ise ortalama sadakat sayısında görünmez. Bir sonraki bölüm bu noktayı açar.

## 3. Zafiyetin kaynağı: Sadakat, hatanın nerede olduğunu söylemez

**Adaletsizliği** makale demografik parite farkıyla ölçer. *s* ∈ {0, 1} hassas özellik (örneğin cinsiyet), *ŷ* modelin pozitif karar verdiği durum olsun:

$$
\varepsilon(M) = \left|\, P(\hat y = 1 \mid s = 1) - P(\hat y = 1 \mid s = 0) \,\right|
$$

Bu değer 0'a yakınsa model iki gruba benzer oranda olumlu karar veriyor demektir. Kara kutunun adaletsizliği ε(*B*) olsun. Makalenin gözlemi basit ama güçlüdür:

- Vekil modelin sadakati %90 ise, 100 örneğin 10'unda kara kutudan farklı karar verir.
- Bu 10 örnek **rastgele seçilmek zorunda değildir**. Hatalar, kara kutunun grup farkını yaratan örneklere (örneğin tam olarak "kadın ve reddedilen" örneklere) yığılırsa, vekil modelin kararları iki grup arasında çok daha dengeli çıkar.
- Sonuç: Sadakat hâlâ %90'dır ama vekilin adaletsizliği ε(*I*) ≪ ε(*B*) olur.

Bu yüzden zafiyet bir "yöntem hatası" değil, **sadakatin neyi ölçmediğiyle** ilgilidir. Denetçi sadakate bakıp memnun olur, vekilin hatalarının nerede toplandığına bakmaz.

> **Benzetme:** Bir öğretmenin öğrenci notlarını açıklamasını düşünün. Açıklamayı yazan kişi, notların %90'ını doğru tahmin eden bir tablo verebilir. Yanlış tahmin ettiği %10'luk kısım tam da ayrımcılık yapılan öğrencilere denk geliyorsa, tablo "adil" görünür ama gerçek notlar adil değildir. Doğruluk oranı hile yapılan yeri göstermez.

## 4. Tehdit modeli ve LaundryML

### 4.1 Oyuncular ve varsayımlar

- **Saldırgan (model sahibi):** Adaletsiz bir kara kutu *B* kullanıyor. Denetçiye göstereceği bir açıklamayı kendisi seçebiliyor ve bu açıklamayı üretmek için kara kutuya sorgu atabiliyor.
- **Denetçi:** Kara kutuya yalnızca dışarıdan bakıyor. Elinde bir "suing group" var: kara kutunun kararlarının etiketlendiği bir veri kümesi (örneğin dava açan kişilerin kayıtları). Açıklamanın sadakatini ve adaletini bu küme üzerinde değerlendiriyor.
- **Saldırganın hedefi:** Vekil modelin (1) suing group'ta kara kutuya yüksek sadakat göstermesi, (2) kara kutudan çok daha adil görünmesi.

### 4.2 Amaç fonksiyonu

Saldırgan, tek bir vekil model üretmek yerine **çok sayıda aday vekil model** üretir ve içlerinden seçer. Her aday için makalenin kullandığı amaç şu biçimdedir (özet):

$$
\min_{I}\;\; (1-\beta)\cdot \text{misc}(I, B) \;+\; \beta\cdot \varepsilon(I) \;+\; \lambda\cdot K(I)
$$

- **misc**(*I*, *B*): vekilin kara kutudan farklı karar verdiği örnek oranı (1 − sadakat),
- **ε**(*I*): vekilin adaletsizliği,
- **K**(*I*): modelin karmaşıklığı (kural listesinde kural sayısı),
- **β** ∈ [0, 1]: adalete verilen ağırlık. β = 0 ise sıradan bir vekil model eğitilir; β arttıkça model sadakatten bir miktar feda edip adalete doğru çekilir,
- **λ**: karmaşıklık cezası.

Makale bu işlemi **LaundryML** adıyla sunar. Kural listeleri **CORELS** algoritmasıyla numaralandırılır, yani β ve λ'nın farklı değerleri için çok sayıda aday kural listesi üretilir.

### 4.3 Seçim kuralı

Yazarlar adayları iki eksende değerlendirir: x ekseninde adaletsizlik, y ekseninde sadakat. Bir saldırgan şu kuralı uygular: **Kara kutudan en az iki kat daha adil olan adaylar arasında sadakati en yüksek olanı seç.** Bu kural, "denetçiye göstereceğim açıklama hem iyi taklit etmeli hem de belirgin biçimde daha adil görünmeli" hedefini formüle eder.

### 4.4 Neden bu kadar kolay?

- Bir kara kutuyu aynı sadakatle taklit eden **birden çok** farklı vekil model vardır (literatürde "Rashomon etkisi" denir). Bunların bir kısmı hassas özelliği kullanır, bir kısmı kullanmaz.
- Saldırgan bu çeşitlilikten yararlanıp "en uygun görüneni" seçer.
- Denetçi bu çeşitliliği görmez; yalnızca seçilen tek bir vekili görür.

## 5. Deney düzeni

### 5.1 Veri setleri ve modeller

| Veri seti | Bağlam | Hassas özellik |
|---|---|---|
| Adult (UCI) | Gelirin yılda 50.000 doların üzerinde olup olmadığı | Cinsiyet |
| ProPublica (COMPAS) | Ceza adaletinde suç tekrarı riski | Irk |

Kara kutu olarak makalenin ana deneyinde rastgele orman kullanılır. Veri; eğitim, suing group ve test olmak üzere üçe bölünür. Kara kutu eğitim verisiyle eğitilir. Aklama, kara kutunun **suing group'u etiketlemesiyle** yapılır; sadakat ve adaletsizlik suing group'ta ölçülür. Test kümesi, aklamanın görmediği veride de geçerli olup olmadığını (genelleme) gösterir.

### 5.2 Değerlendirme ölçütleri

- **Sadakat:** Vekil ile kara kutunun aynı kararı verdiği örnek oranı.
- **Adaletsizlik:** Demografik parite farkı ε.
- **Genelleme:** Aynı ölçütlerin test kümesinde ne olduğu. Aklanmış model suing group'ta adil görünüp testte görünmüyorsa, aklama yalnızca o kümeye özgüdür.

## 6. Sonuçlar ve yorumları

### 6.1 Ana sonuçlar

Makalenin Adult üzerindeki rastgele orman deneyi için raporladığı değerler:

| Ölçüt | Makale (Adult) |
|---|---|
| Kara kutu doğruluğu (eğitim / suing / test) | 0,852 / 0,843 / 0,849 |
| Kara kutu adaletsizliği (suing / test) | 0,13 / 0,18 |
| Seçilen aklanmış modelin sadakati | 0,908 |
| Seçilen aklanmış modelin adaletsizliği | 0,058 |

ProPublica için kara kutu doğruluğu 0,714 / 0,659 / 0,676 olarak verilir; ayrıntılar makalenin tablolarındadır.

**Yorum.** Adaletsizliği kara kutunun yarısından aşağıya çeken, ama sadakati yüzde 90'ın üzerinde tutan bir vekil model bulunabiliyor. Yani "yüksek sadakat + düşük adaletsizlik" birlikte mümkün. Bir denetçi yalnızca bu vekile baktığında, kara kutunun adil olduğu sonucuna varır.

### 6.2 Genelleme

Makale, aklamanın suing group dışında (test kümesinde) da büyük ölçüde korunduğunu gösterir. Bu önemlidir: Saldırgan açıklamayı yalnızca bir veri kümesine göre özel olarak kurgulamış olmuyor, açıklama yeni veride de adil görünmeyi sürdürüyor. Denetçi başka bir örneklemle kontrol etse bile aynı izlenimi alır.

### 6.3 Bu bir "saldırı" mı, yoksa bir "risk" mi?

Makalenin başlığı bu ayrımı yansıtır: *the risk of rationalization*. Saldırıyı gerçekleştirmek için kara kutunun içine dokunmak gerekmez. Yalnızca kara kutunun kararlarına erişim ve bir aday modeller havuzu yeter. Bu yüzden asıl çıkarım, **açıklamaların tek başına bir adalet kanıtı olarak kullanılamayacağıdır**.

## 7. Uzantı: Aklama tespit edilebilir mi? (FRAUD-Detect)

Shamsabadi ve ark. (NeurIPS 2022, *Washing the Unwashable: On the (Im)possibility of Fairwashing Detection*) bu sorunun savunma tarafını inceler. FRAUD-Detect'in fikri şudur: Aklanmış bir vekilin hataları rastgele dağılmaz, gruplara göre **asimetrik** dağılır.

**Algoritmanın özeti (Algoritma 1).** Her hassas alt grup *a* için, kara kutuyu referans alarak vekil modelin hata/başarı oranlarından oluşan bir vektör hesaplanır:

$$
C_a = [\,\mathrm{TPR},\ \mathrm{FPR},\ \mathrm{TNR},\ \mathrm{FNR}\,]
$$

İki alt grubun vektörleri arasındaki ayrışma KL ıraksaklığıyla ölçülür:

$$
C_{KL} = D_{KL}\!\left(C_0 \,\|\, C_1\right)
$$

*C*<sub>KL</sub> belirli bir eşik τ'yu aşarsa "bu vekil aklanmış olabilir" uyarısı verilir. Dürüst bir vekilde hatalar iki grupta benzer dağıldığı için *C*<sub>KL</sub> küçüktür.

**Benim uygulamam.** Vektörü 2'ye bölüp dağılıma çeviriyor ve küçük alt gruplarda oranlar oynak olduğu için Laplace düzeltmesi ekliyorum. Eşik τ'yu makaledeki Ek G'den almıyorum; dürüst vekillerin bootstrap dağılımının 95. yüzdeliği olarak **kendim kalibre ediyorum**. Bu yüzden sonuçlar makalenin raporladığı tespit oranlarıyla doğrudan karşılaştırılamaz.

## 8. Makalenin konumu, sınırları ve sonrası

### 8.1 Literatürdeki yeri

- **Açıklamaya saldırılar:** Slack ve ark. (2020), *Fooling LIME and SHAP*, LIME ve SHAP'ı kandırır: kara kutu, açıklama yönteminin ürettiği yapay noktaları tanıyıp o noktalarda farklı davranır. Fairwashing ise **global vekil modeli** kandırır ve kara kutunun hiçbir davranışını değiştirmez. İki çalışma aynı soruyu iki farklı açıklama türü için sorar: "Açıklama modelin gerçek davranışını yansıtıyor mu?"
- **Adalet ve yorumlanabilirlik:** Adalet denetimlerinin açıklamalara dayanması, açıklamanın güvenilirliği sorusunu doğrudan düzenleme (regülasyon) konusuna bağlar.
- **Takip çalışmaları:** Aynı grup, aklamanın ne kadar yaygın olabileceğini ve nasıl azaltılabileceğini daha sonraki çalışmalarda inceler (ör. *Characterizing the risk of fairwashing*, NeurIPS 2021). Burada ayrıntılarına girmiyorum.

### 8.2 Yazarların vardığı sonuç

Yazarlara göre yorumlanabilir bir vekil model, kara kutunun adaletini **tek başına** kanıtlayamaz. Denetim süreçlerinde açıklamanın üretildiği yöntemin ve açıklamanın doğrulanma biçiminin de denetlenmesi gerekir.

### 8.3 Eleştirel değerlendirme (kendi yorumum)

- **Saldırgan bilgisi:** Saldırgan kara kutunun kararlarını suing group için serbestçe üretebiliyor ve denetçinin hangi kümede ölçeceğini biliyor. Denetçi açıklamayı bağımsız bir kümede doğrularsa bu avantaj azalır (testteki genelleme bunun ne kadar kalıcı olduğunu gösterir).
- **Tek ölçüt:** Makale demografik pariteyi ele alır. Eşitlenmiş olasılık (equalized odds) gibi hata temelli ölçütlerde vekilin hatalarının yığıldığı bölge daha kolay yakalanabilir.
- **Küçük veride etki:** Suing group çok küçükse hem aklama hem de sadakat/adaletsizlik ölçümü gürültülü hale gelir.
- **Savunma yönleri:** Genel sadakat yerine **alt grup bazlı** sadakata bakmak, hata yönünü (hangi sınıfa kaydığını) incelemek, açıklamayı suing group dışında bir kümede doğrulamak ve FRAUD-Detect gibi tespit yöntemlerini kullanmak.

## 9. Kendi yeniden üretimimde gözlenenler

Saldırıyı ve tespiti kendi kodumla yeniden uyguladım (`analiz.ipynb`; veri: `veri.csv`). Makaleden farklarım: LaundryML'in CORELS kural listesi yerine **sığ karar ağaçlarından oluşan bir aday havuzu** (derinlik 2-8, hassas sütun açık/kapalı, "adaletsizliği besleyen hücrelere" ağırlık azaltma) kullandım; veri bölme oranlarını kendim seçtim (makalede yok); FairML yerine permütasyon önemi kullandım. Bu yüzden sayılar makaledekiyle birebir aynı çıkmaz, karşılaştırılan şey bulgunun niteliğidir.

### 9.1 Orijinal veri (Adult, 45.222 kayıt, 5 farklı bölme)

Hassas özellik cinsiyet (`sex_Male`), pozitif sınıf oranı 0,248. Kara kutu rastgele orman. Veri bölmesi: %50 eğitim, %25 suing group (11.305 örnek), %25 test.

| Ölçüm | Makale (Adult) | Bu yeniden üretim (5 bölme, ort. ± std) | Fark |
|---|---|---|---|
| Kara kutu doğruluğu (test) | 0,849 | 0,842 ± 0,002 | −0,007 |
| Kara kutu adaletsizliği, suing | 0,130 | 0,189 ± 0,011 | +0,059 |
| Kara kutu adaletsizliği, test | 0,180 | 0,196 ± 0,010 | +0,016 |
| Seçilen modelin sadakati (suing) | 0,908 | 0,887 ± 0,005 | −0,021 |
| Seçilen modelin adaletsizliği (suing) | 0,058 | 0,058 ± 0,016 | 0,000 |

- **Ana bulgu yeniden üretildi.** "En az 2 kat daha adil" kuralı 5/5 bölmede sağlandı. Seçilen aklanmış model, kara kutunun adaletsizliğini yaklaşık 0,19'dan 0,06'ya indirirken sadakati %89 civarında tuttu. Seçilen modelin adaletsizliği makaledeki değerle aynı çıktı (0,058).
- **Farklar.** Kara kutunun adaletsizliği benim kurulumumda makaledekinden yüksek (suing grupta 0,189'a karşı 0,13), sadakat ise biraz düşük (0,887'ye karşı 0,908). Bunun olası nedenleri: veri bölme oranlarını makale vermediği için kendim seçmem, rastgele ormanın ayarları ve LaundryML yerine ağaç havuzu kullanmam. Sonuçların niteliği aynı, sayılar birebir aynı değil.
- **Gerçek aklama.** 5 bölmenin 5'inde seçilen model, aklama olmadan da zaten adil çıkan bir dürüst vekil değildi. Yani Adult'ta aklama gerçekten adaletsizliği gizliyor (German'da bu oran 2/5).
- **Model boyutu.** Seçilen aklanmış ağaçlar büyük: 56 ile 71 yaprak. German'da bu sayı 8-14. Yani Adult'ta sadakati korumak için çok daha karmaşık bir vekil gerekti ve bu "okunabilir bir açıklama" kavramını zorluyor.
- **Hassas özelliğin sırası (permütasyon önemi, 104 özellik).** `sex_Female` kara kutuda 22., dürüst vekilde 12., aklanmış vekilde 16. sırada; `sex_Male` sırasıyla 21., 18. ve 18. Cinsiyetin sırası aklamayla belirgin biçimde kaybolmuyor, yani bu ölçütle aklama "hassas özelliği listeden silmiyor". Aklanmış vekil, kara kutunun grup farkını başka özelliklerle taklit ediyor.

### 9.2 Kendi verim (Kaggle German Credit, 1000 başvuru)

Hassas özellik cinsiyet (erkek %69), etiket Risk = good (%70). Veri bölmesi: %40 eğitim, %30 suing group (300 örnek), %30 test. 5 farklı rastgele bölmeyle çalıştırdım.

| Ölçüm | Değer (5 bölme) |
|---|---|
| Kara kutunun adaletsizliği (suing group) | 0,088 ± 0,040 |
| Seçilen modelin sadakati | 0,905 ± 0,009 |
| Seçilen modelin adaletsizliği | 0,025 ± 0,012 |
| "En az 2 kat daha adil" kuralı sağlanan bölme | 5 / 5 |
| Çoğunluk sınıfını sabit tahmin eden basit modelin sadakati | 0,797 ± 0,036 |
| Seçilen modelin gerçekten "aklanmış" (dürüst vekil zaten adil değil) olduğu bölme | 2 / 5 |

- **Aklama bu veride de çalıştı.** Küçük bir suing group'ta (300 örnek) bile sadakat %90 civarında kalırken adaletsizlik yaklaşık üçte bire indi. Seçilen modelin sadakati, sabit tahmin tabanının (0,797) belirgin biçimde üzerinde; yani sonuç "kara kutu zaten hep aynı sınıfı veriyor" kolaylığından kaynaklanmıyor.
- **Önemli nüans: kara kutu bu veride zaten az adaletsiz.** Adaletsizlik 0,088 ile Adult'a göre düşük. 5 bölmenin 3'ünde sıradan (aklamasız) dürüst vekil de zaten 2 kat kuralını sağlıyor; yani orada ortada gizlenecek büyük bir adaletsizlik yok. Aklamanın gerçekten bir şey **gizlediği** bölme sayısı 2/5. Bu sonuç makalenin iddiasını zayıflatmaz ama saldırının etkisinin kara kutunun başlangıç adaletsizliğine bağlı olduğunu gösterir.
- **Hassas özelliğin sırası.** Permütasyon önemiyle 25 özellik arasında cinsiyetin sırası kara kutuda ve dürüst vekilde 9, aklanmış vekilde 11 çıktı. Düşüş küçük; aklanmış modelin hassas sütunu kullanmadığı tek müşterilik demoda da görülüyor.
- **Canlı demo.** 24 yaşında, 1275 kredi tutarı, 15 ay vadeli bir kadın başvuru için kara kutu "bad" diyor. Dürüst vekil (10 yaprak, hassas sütun açık) "good" der, çünkü kural yolunda `Credit amount > 1239.5` var. Aklanmış vekil (14 yaprak, hassas sütun kapalı) aynı müşteriyi `Credit amount <= 1316.5` ve `Age <= 36.5` kuralları üzerinden "bad" olarak kara kutuyla uyumlu sınıflandırıyor. Açıklama, cinsiyet kullanmadan ve makul görünen kurallarla kara kutunun kararını verir.
- **Genelleme.** Suing group'u 75 örneğe düşürdüğümde suing'de adaletsizlik 0,075, testte 0,022 çıktı; tam boyutta (300) 0,029 ve 0,014. Aklama suing group'a özgü kalmadı, testte de adil görünüyor.
- **Karmaşıklık.** Ağaç derinliği 2'den 8'e çıkarken sadakat 0,843'ten 0,920'ye çıktı; adaletsizlik 0,013 ile 0,029 arasında kaldı. Daha karmaşık model daha sadık ama adaleti bozmuyor.

### 9.3 Uzantı: FRAUD-Detect bu aklamayı yakalıyor mu?

- **Adult'ta tespit çalıştı, German'da çalışmadı.** Adult'ta FRAUD-Detect seçilen aklanmış modeli 5 bölmenin 5'inde yakaladı (*C*<sub>KL</sub> 0,065 ile 0,121, τ 0,021 ile 0,028). German'da 5 bölmenin hiçbirinde yakalayamadı (örneğin bir bölmede *C*<sub>KL</sub> = 0,047, τ = 0,052; sınıra çok yakın). German'da aday havuzunda hata asimetrisi ε ile *C*<sub>KL</sub> arasındaki Spearman korelasyonu −0,83 çıktı: beklenenin tersi yönde.
- **Neden fark var?** Adult'ta suing group 11.305, German'da 300 örnek. Büyük örneklemde alt grup oranları kararlı, τ düşük ve ayrım güvenilir. Küçük örneklemde oranlar oynak, τ yüksek ve belirsiz. Bu nedenle tespitin başarısı hem aklamanın yapısına hem de örnek sayısına bağlı.
- **Dedektör kontrollü deneyde çalışıyor.** Kara kutunun kararlarını, adaletsizliği besleyen hücrelerde belirli bir olasılıkla (q) çevirerek "ideal aklama" ürettim; karşılaştırma olarak aynı oranda gruptan bağımsız gürültü ekledim. q = 0,20'de Adult'ta *C*<sub>KL</sub> aklamada 0,77, gürültüde 0,001 (τ = 0,02); German'da aklamada 0,38, gürültüde 0,015 (τ = 0,05). Dedektör kasıtlı ve yönlü bir aklamayı gürültüden ayırabiliyor. German'da benim ağaç havuzumdaki aklama böyle yönlü bir hata üretmediği ve örnek sayısı küçük olduğu için yakalanmıyor.
- **Okuma.** Tespit, aklamanın **yapısına** bağlı. FRAUD-Detect'in varsayımı (hataların gruplara göre asimetrik olması) bu veride ve bu aday havuzunda sağlanmıyor. Bu, tespitin her koşulda güvenilir olmadığını gösterir; makale de bilgili bir saldırganın tespiti dolanabileceğini söyler. Bulgular benim kendi τ kalibrasyonuma ve suing group boyutuna duyarlıdır.

## 10. Terim sözlüğü

| İngilizce | Türkçe | Kısa açıklama |
|---|---|---|
| Global surrogate | Genel vekil model | Kara kutunun kararlarını taklit etmek için eğitilen yorumlanabilir model |
| Fidelity | Sadakat | Vekil ile kara kutunun aynı kararı verme oranı |
| Fairwashing | Adalet aklama | Adaletsiz bir modeli, açıklamayla adil gösterme |
| Rationalization | Gerekçelendirme | Aklamanın makaledeki genel adı |
| Suing group | Dava grubu | Kara kutunun etiketlediği ve açıklamanın değerlendirildiği küme |
| Demographic parity | Demografik parite | Gruplar arası olumlu karar oranı farkının 0 olması |
| Unfairness | Adaletsizlik | Demografik parite farkı ε |
| LaundryML | LaundryML | Makalenin aday vekil modeller üretip seçen yöntemi |
| CORELS | CORELS | Optimal kural listesi öğrenen algoritma |
| Rashomon effect | Rashomon etkisi | Aynı veriyi eşit iyi açıklayan birçok farklı model olması |
| Model enumeration | Model numaralandırma | Çok sayıda aday modelin sistematik üretimi |
| Equalized odds | Eşitlenmiş olasılık | Gruplar arası TPR ve FPR eşitliği |
| FRAUD-Detect | FRAUD-Detect | Alt grup hata oranlarının KL ayrışmasıyla aklama tespiti |
| Bootstrap | Önyükleme | Veriden yerine koyarak yeniden örnekleme |
| Permutation importance | Permütasyon önemi | Bir özelliği karıştırınca doğruluğun ne kadar düştüğüyle ölçülen önem |

---

Kaynak: U. Aïvodji, H. Arai, O. Fortineau, S. Gambs, S. Hara, A. Tapp. "Fairwashing: the risk of rationalization." ICML 2019. arXiv:1901.09749 · Orijinal kod: github.com/aivodji/LaundryML

Diğer anılan çalışmalar: Shamsabadi ve ark. 2022, "Washing the Unwashable: On the (Im)possibility of Fairwashing Detection" (NeurIPS); Slack ve ark. 2020, "Fooling LIME and SHAP" (AIES); Molnar, "Interpretable Machine Learning", Bölüm 25.
