# Açıklamalar Kandırılabilir mi?

**Slack ve ark. (2020), *Fooling LIME and SHAP* üzerine ayrıntılı Türkçe anlatım**

Makale: D. Slack, S. Hilgard, E. Jia, S. Singh, H. Lakkaraju. AIES 2020. doi:10.1145/3375627.3375830 · arXiv:1911.02508  
Hazırlayan: İlker Uğur Kaya · Açıklanabilir Yapay Zeka dersi, makale yeniden üretimi

> Bu belge makalenin çevirisi değildir. Makalenin akışını izleyerek fikirlerini, yöntemini ve deneylerini kendi anlatımımla ve ek açıklamalarla aktarır. Yeniden üretim bulguları (Bölüm 9) ve terim sözlüğü (Bölüm 10) bana aittir. Kesin ifadeler ve şekiller için orijinal makaleye başvurulmalıdır.


**İçindekiler**

1. Problem: Açıklamaya neden güveniyoruz?
2. Arka plan: LIME nasıl çalışır?
3. Arka plan: Kernel SHAP nasıl çalışır?
4. Zafiyetin kaynağı: Dağılım dışı örnekler
5. Tehdit modeli ve iskele saldırısı
6. Deney düzeni
7. Sonuçlar ve yorumları
8. Makalenin konumu, sınırları ve sonrası
9. Kendi yeniden üretimimde gözlenenler
10. Terim sözlüğü

## 1. Problem: Açıklamaya neden güveniyoruz?

Kredi onayı, suç tekrarı riski ya da tıbbi tanı gibi alanlarda kullanılan modellerin çoğu karmaşık ve çoğu zaman ticari olarak kapalıdır. Bir denetçi, düzenleyici kurum ya da alan uzmanı bu modelin içine bakamaz. Bakabilse bile milyonlarca parametreden anlamlı bir sonuç çıkaramaz. Bu boşluğu dolduran araçlar **sonradan (post hoc) açıklama yöntemleridir**: Modeli değiştirmeden, yalnızca girdi verip çıktı alarak "bu karar hangi özelliklere dayanıyor?" sorusunu cevaplamaya çalışırlar.

Bu araçların en yaygın kullanım amaçlarından biri **önyargı denetimidir**. Bir denetçi, modelin kararlarını açıklatır ve ırk, cinsiyet gibi hassas özelliklerin öne çıkıp çıkmadığına bakar. Çıkmıyorsa modeli "makul" kabul eder. Bu iş akışının gizli bir varsayımı vardır: **Açıklama, modelin gerçek davranışını yansıtır.**

Makalenin sorduğu soru bu varsayımın kırılgan olup olmadığıdır. Yazarların cevabı açık: Model sahibi kötü niyetliyse ve denetçinin LIME ya da SHAP kullanacağını biliyorsa, açıklamayı istediği gibi şekillendirebilir. Model gerçek müşterilere ayrımcılık yapmaya devam ederken, açıklama tamamen masum bir tablo çizer. Bu yüzden makale yalnızca bir "saldırı" önermiyor, aynı zamanda açıklama tabanlı denetim pratiğinin güvenilirliğini sorguluyor.

> **Tek cümlelik özet:** LIME ve SHAP modeli gerçek veri noktalarında değil, kendi ürettikleri yapay noktalarda sorgular. Bu noktalar gerçek veriden ayırt edilebildiği için model, "denetlendiğini" anlayıp o anda farklı davranabilir.

## 2. Arka plan: LIME nasıl çalışır?

LIME'ın (Ribeiro ve ark., 2016) temel fikri şudur: Karmaşık bir model *f* genel olarak doğrusal olmasa da, tek bir örnek *x*'in çok yakın çevresinde basit bir modelle yaklaşık olarak temsil edilebilir. LIME bu yerel modeli, genellikle doğrusal bir *g*'yi, şu amaçla bulur:

$$
\xi(x) = \arg\min_{g \in G} \; L(f, g, \pi_x) + \Omega(g)
$$

Burada *L*, *g*'nin *f*'yi *x*'in çevresinde ne kadar kötü taklit ettiğini ölçer. π<sub>*x*</sub>, bir noktanın *x*'e ne kadar yakın olduğuna göre verilen ağırlıktır. Ω(*g*) ise açıklamanın karmaşıklığına verilen cezadır (örneğin kullanılan özellik sayısı). Uygulamada bu problem şu adımlarla çözülür:

1.  **Örnekleme:** *x*'in çevresinde çok sayıda yapay nokta *z* üretilir (varsayılan 5000).
2.  **Sorgulama:** Her nokta kara kutu modele verilir ve *f*(*z*) kaydedilir.
3.  **Ağırlıklandırma:** Her noktaya *x*'e uzaklığına göre bir ağırlık verilir. Tablo verisinde varsayılan çekirdek, uzaklık *d* için exp(−*d*²/σ²) biçimindedir (kütüphane bunun karekökünü kullanır). Varsayılan genişlik σ = 0.75·√*M*, burada *M* özellik sayısıdır.
4.  **Yerel model:** Bu ağırlıklarla bir ağırlıklı doğrusal regresyon (ridge) kurulur. Katsayılar açıklama olarak sunulur.

### Tablo verisinde örnekleme ayrıntısı (saldırının anahtarı)

Görüntü ya da metinde LIME, girdinin parçalarını açıp kapatarak örnek üretir. Tablo verisinde ise durum farklıdır ve saldırıyı mümkün kılan da budur. `lime_tabular`'ın varsayılan davranışında:

- Sayısal özellikler, eğitim verisindeki ortalama ve standart sapmaya göre **her özellik için ayrı ayrı** normal dağılımdan çekilir.
- Kategorik özellikler, eğitim verisindeki kategori sıklıklarına göre yine **birbirinden bağımsız** çekilir.

Bu bağımsız örnekleme, özellikler arasındaki ilişkileri tamamen yok sayar. Gerçek veride 22 yaşında birinin 30 yıllık iş deneyimi olamaz, kredi süresi ile tutarı birlikte artma eğilimindedir, bazı değişkenler yalnızca tam sayı değer alır. LIME'ın ürettiği noktalar ise bu kuralları bilmez. Sonuç: Üretilen noktaların önemli bir kısmı gerçek hayatta karşılaşılmayacak, **veri manifoldunun dışında** kalan noktalardır.

## 3. Arka plan: Kernel SHAP nasıl çalışır?

SHAP (Lundberg ve Lee, 2017) açıklamayı oyun teorisindeki **Shapley değerlerine** dayandırır. Özellikleri bir oyunun oyuncuları, modelin tahminini de oyunun kazancı gibi düşünür. Bir özelliğin katkısı, onun olası tüm özellik koalisyonlarına eklendiğinde yarattığı ortalama farktır:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!} \left[ v(S \cup \{i\}) - v(S) \right]
$$

Burada *F* tüm özellikler kümesi, *v*(*S*) ise yalnızca *S*'deki özellikler "biliniyorken" modelin beklenen çıktısıdır. Pratikte bir özelliğin "bilinmemesi", o özelliğin değerinin bir **arka plan dağılımından** gelen değerlerle değiştirilmesi demektir. Makalede arka plan olarak eğitim verisinin 10 kümeli k-means merkezleri kullanılır.

**Kernel SHAP**, bu toplamı tüm koalisyonlar üzerinden hesaplamak yerine koalisyonları örnekler ve özel bir ağırlıklandırmayla LIME'a benzer bir ağırlıklı doğrusal regresyon çözer. Dolayısıyla SHAP da LIME gibi modeli **yapay, karışık noktalarda** sorgular. Örneğin bir müşterinin yaşı ve cinsiyeti korunup diğer tüm özellikleri bir küme merkezinden alınabilir.

SHAP'ın LIME'dan ayrılan önemli bir özelliği **yerel doğruluktur (local accuracy)**:

$$
\phi_0 + \sum_i \phi_i = f(x)
$$

Yani katkıların toplamı, tahminin arka plan ortalamasından farkına tam olarak eşit olmak zorundadır. Bu kısıt, saldırıya karşı SHAP'ı kısmen dirençli yapan özelliktir (bkz. Bölüm 7).

## 4. Zafiyetin kaynağı: Dağılım dışı örnekler

Makalenin ilk gözlemi basit ama güçlüdür. Yazarlar COMPAS verisini ve LIME tarzı bozulmuş kopyalarını birlikte PCA ile iki boyuta indirir (makalenin Şekil 1'i). İki nokta bulutu belirgin biçimde farklı yerlere düşer. Gözle bile görülebilen bu fark, basit bir sınıflandırıcının "bu nokta gerçek mi yoksa bir açıklama yönteminin ürettiği yapay nokta mı?" sorusunu yüksek doğrulukla cevaplayabileceğini gösterir.

Bu noktada durum şöyle özetlenebilir: Açıklama yöntemi modele "bu bölgede nasıl davranıyorsun?" diye sorar, ama soruyu gerçekçi olmayan noktalar üzerinden sorar. Model bu noktaları tanıyabiliyorsa, gerçek verideki davranışından bağımsız bir cevap verebilir. Açıklama yöntemi bu cevabı modelin gerçek davranışı sanır.

> **Benzetme:** Bir sürücünün hız sınırına uyup uymadığını yalnızca radar levhasının hemen önünde ölçtüğünüzü düşünün. Levhayı tanıyan sürücü yalnızca orada yavaşlar. Ölçüm doğru çalışıyordur, ama ölçtüğü şey sürücünün gerçek davranışı değildir.

## 5. Tehdit modeli ve iskele saldırısı

### 5.1 Oyuncular ve varsayımlar

- **Saldırgan (model sahibi):** Taraflı bir model *f*'yi kullanıma almak istiyor. Bu model örneğin yalnızca ırka göre karar veriyor. Saldırgan, denetçinin LIME ya da SHAP kullanacağını biliyor ve eğitim verisine erişimi var.
- **Denetçi:** Modele yalnızca kara kutu olarak erişiyor (girdi verip çıktı alıyor). Modelin önyargılı olup olmadığına açıklamalara bakarak karar veriyor.
- **Saldırganın hedefi:** (1) Gerçek veride *f* gibi davranmak, yani önyargıdan vazgeçmemek. (2) Açıklamalarda hassas özelliğin görünmemesini sağlamak.

### 5.2 İskele (scaffolding) model

Saldırgan, *f*'nin etrafına bir "iskele" kurar. Bu iskele üç parçadan oluşur: taraflı model *f*, masum görünen bir model *ψ* ve bir dağılım dışı (OOD) dedektörü. Ortaya çıkan saldırgan model *e* şöyle çalışır:

$$
e(x) = \begin{cases} f(x) & \text{dedektör } x\text{'i gerçek veri sayarsa} \\ \psi(x) & \text{dedektör } x\text{'i yapay nokta sayarsa} \end{cases}
$$

*ψ*, hassas özellikle ilişkisiz özelliklere dayanır. Böylece LIME'ın ya da SHAP'ın sorguladığı yapay noktaların neredeyse tamamında cevabı *ψ* verir ve açıklama *ψ*'nin mantığını yansıtır. Gerçek müşteriler ise dedektör tarafından "gerçek" sayıldıkları için *f* ile değerlendirilir.

### 5.3 Neden açıklama ψ'yi gösterir? (Sezgisel analiz)

LIME'ın yerel regresyonuna giren 5000 noktadan yalnızca biri (açıklanan örneğin kendisi) kesin olarak gerçektir. Geri kalanlar bağımsız örneklemeyle üretildiği için dedektör onları büyük oranda "yapay" olarak işaretler. Bu yüzden regresyonun gördüğü etiketlerin neredeyse tamamı *ψ*'den gelir. Doğrusal model de doğal olarak *ψ*'nin baktığı özelliğe büyük, hassas özelliğe ise sıfıra yakın bir katsayı verir. Açıklanan örnek en yüksek ağırlığı alsa bile binlerce noktanın içinde tek başına sonucu değiştiremez.

### 5.4 OOD dedektörünün eğitimi

Dedektör, basit bir ikili sınıflandırma problemi olarak eğitilir:

1.  Eğitim verisi *X* alınır ve her örnek bozularak yapay bir küme *X*<sub>p</sub> üretilir. LIME için bozma, özelliklere rastgele gürültü eklemektir. SHAP için ise bazı özelliklerin arka plan değerleriyle değiştirilmesidir. Yani bozma, saldırılan yöntemin örnekleme biçimini taklit eder.
2.  *X*'teki örnekler "gerçek", *X*<sub>p</sub>'dekiler "yapay" diye etiketlenir.
3.  *X* ∪ *X*<sub>p</sub> üzerinde bir sınıflandırıcı eğitilir. Makalede bu, 100 ağaçlı bir random forest'tır.

> **Uygulama ayrıntıları (orijinal koddan):** LIME saldırısında gürültünün standart sapması 0.3'tür ve her gerçek örnek için 30 bozulmuş kopya üretilir. Makale metninde gürültü 𝒩(0, 1) olarak anılır; aradaki fark savunmada sorulabilir. Ayrıca LIME saldırısında dedektör yalnızca sayısal sütunlara bakar, kategorik sütunlar dedektöre girmez.

### 5.5 ψ nasıl seçilir?

- **Sentetik ilişkisiz özellik (COMPAS, CC):** Veriye hassas özellikle korelasyonu sıfır olan rastgele bir sütun eklenir. *ψ* bu sütuna göre karar verir. Denetçi açıklamada anlamsız bir özellik görür ama bu en azından ayrımcı görünmez.
- **İki ilişkisiz özellik ve XOR:** *ψ* iki sentetik sütunun XOR'una göre karar verir. Burada amaç tek bir özelliğin açıklamayı tek başına domine etmemesi, önemin iki özelliğe paylaştırılmasıdır.
- **Mevcut gerçek bir özellik (German):** Yapay sütun eklemek yerine verideki makul bir özellik kullanılır: kredi taksitinin gelire oranı. Bu senaryo daha gerçekçi ve daha tehlikelidir, çünkü açıklama yalnızca masum değil, aynı zamanda **inandırıcıdır**. Bir denetçi "model taksit/gelir oranına bakıyor" açıklamasını büyük ihtimalle kabul eder.

## 6. Deney düzeni

### 6.1 Veri setleri

| Veri seti                   | Bağlam                                            | Hassas özellik                                      | Etiket                                |
|-----------------------------|---------------------------------------------------|-----------------------------------------------------|---------------------------------------|
| COMPAS (ProPublica)         | Ceza adaleti, suç tekrarı riski                   | Irk                                                 | Yüksek / düşük risk                   |
| Communities and Crime (UCI) | 1994 ABD topluluğunun sosyoekonomik ve suç verisi | Beyaz nüfus yüzdesi (yaklaşık %2.7 ile %99.6 arası) | Şiddet suçu oranı medyanın üstünde mi |
| German Credit (UCI)         | Yaklaşık 1000 kredi başvurusu                     | Cinsiyet (%69 erkek)                                | İyi / kötü müşteri (%70 iyi)          |

### 6.2 Modeller

- ***f*:** Kasıtlı olarak "mükemmel ayrımcı" kurulur. COMPAS ve CC'de yalnızca ırka, German'da yalnızca cinsiyete bakar. Bu aşırı uç seçim bilinçlidir: Önyargı bu kadar açıkken bile gizlenebiliyorsa, daha ince önyargılar haydi haydi gizlenebilir.
- ***ψ*:** Bölüm 5.5'teki seçenekler.
- **Dedektör:** 100 ağaçlı random forest.

### 6.3 Açıklama yöntemleri ve ayarları

- **LIME:** Varsayılan tablo açıklayıcısı, sürekli özellikleri ayrıklaştırmadan.
- **SHAP:** Varsayılan Kernel SHAP, 10 kümeli k-means arka planı.
- **Bölme:** Rastgele %90 eğitim ve %10 test. Saldırı eğitim kümesiyle kurulur, test kümesinde değerlendirilir.

### 6.4 Değerlendirme ölçütleri

- **Sıralama ölçütü:** Her test örneği için özellikler, açıklamadaki katsayının mutlak değerine göre sıralanır. Ardından her özelliğin 1., 2. ve 3. sırada kaç örnekte göründüğü yüzde olarak raporlanır. Saldırı başarılıysa hassas özellik 1. sıradan kaybolur, *ψ*'nin özelliği oraya yerleşir.
- **Sadakat (fidelity):** Test kümesindeki gerçek örneklerde *e*'nin *f* ile aynı kararı verme oranı. Bu oranın yüksek olması, saldırganın gerçek dünyada önyargıdan vazgeçmediğini gösterir. Sadakat düşükse saldırı "önyargıyı gizlemiş" değil, kısmen "önyargıyı kaldırmış" olur.

## 7. Sonuçlar ve yorumları

### 7.1 Ana sonuçlar

| Durum                                                                              | COMPAS | Comm. & Crime | German |
|------------------------------------------------------------------------------------|--------|---------------|--------|
| LIME, tek ilişkisiz özellik: önemin hassas özellikten ψ'nin özelliğine kayma oranı | %100   | %100          | %91    |
| SHAP, tek ilişkisiz özellik                                                        | %84    | %100          | %85    |
| SHAP, iki ilişkisiz özellik: hassas özelliğin 1. sıradan çıkarılma oranı           | %67    | %71           | –      |
| Sadakat, LIME saldırısı                                                            | %100   | %100          | %100   |
| Sadakat, SHAP saldırısı                                                            | %91    | %75           | %80    |

**LIME tamamen kandırılıyor.** Tek ilişkisiz özellikli saldırıda hassas özellik 1. sıradan neredeyse hep kayboluyor. 2. ve 3. sıralarda da diğer sıradan özelliklerden daha sık görünmüyor; yani açıklamaya bakan biri onu "önemsiz özelliklerden biri" olarak algılıyor. Üstelik sadakat %100: Saldırgan gerçek veride hiçbir şeyden ödün vermiyor.

**SHAP kandırılıyor ama daha zor.** Tek özellikli saldırıda başarı yüksek (%84–100), iki özellikli XOR saldırısında belirgin biçimde düşüyor (%67–71). Sadakat de LIME saldırısındakinden düşük (%75–91), yani SHAP'ı kandırmak için saldırgan gerçek veride bir miktar ödün vermek zorunda kalıyor.

### 7.2 SHAP neden daha dirençli?

İki neden öne çıkıyor. Birincisi **yerel doğruluk** kısıtı: Katkıların toplamı tahmin farkına eşit olmak zorunda. *ψ* iki özelliğin XOR'una bakıyorsa, hiçbir tekil özellik bu farkı tek başına açıklayamaz. SHAP katkıyı birden fazla özelliğe yaymak zorunda kalır ve bu yayılma sırasında hassas özellik de pay alabilir. İkincisi **örnekleme biçimi**: SHAP'ın yapay noktaları, bazı özellikleri gerçek değerinde tutup diğerlerini küme merkezlerinden aldığı için gerçek veriye LIME'ın noktalarından daha çok benzer. Bu da dedektörün işini zorlaştırır ve sadakat düşer.

### 7.3 Dedektör ne kadar iyi olmalı? (makalenin Şekil 5'i)

Yazarlar COMPAS'ta farklı doğruluklarda dedektörler eğitip saldırının başarısını dedektörün F1 skoruna göre çizer. Dedektör zayıfken (gerçek ve yapay noktaları ayıramıyorken) iki yöntem de ırkı en önemli özellik olarak bulur, yani saldırı başarısız olur. Yöntemler arasında ilginç bir fark var:

- **LIME:** Daha iyi bir dedektöre ihtiyaç duyar, ama F1 yaklaşık 0.8'i geçtiği anda çok hızlı ve neredeyse tamamen çöker. Geçiş keskindir.
- **SHAP:** Daha zayıf bir dedektörle (F1 ≈ 0.45) bile etkilenmeye başlar, ama etki kademelidir. Önyargıyı tamamen gizlemek için çok isabetli bir dedektör gerekir.

Çıkarım: Saldırının pratikte uygulanabilmesi için gereken tek koşul, yapay noktaların gerçek noktalardan "makul" doğrulukla ayırt edilebilmesidir. Bağımsız örnekleme yapan yöntemlerde bu koşul kolayca sağlanır.

### 7.4 Hiperparametrelere dayanıklılık

- **Yöntemler arası aktarım:** LIME'a göre kurulmuş bir saldırı SHAP'ı kandıramıyor, çünkü dedektör LIME'ın gürültüsünü tanımak üzere eğitilmiş, SHAP'ın karışık noktalarını tanımıyor. SHAP'a göre kurulmuş saldırının LIME üzerindeki etkisi ise değişken. Bu, saldırganın denetçinin hangi yöntemi kullanacağını bilmesinin önemli olduğunu gösteriyor.
- **LIME çekirdek genişliği:** Denetçi çekirdeği 0.15·√*M* ile 0.95·√*M* arasında değiştirse de, ya da uzaklık ölçüsünü ℓ<sub>1</sub>'e çevirse de saldırı aynı derecede etkili kalıyor. Yani denetçinin ayarlarla oynayarak saldırıyı fark etmesi kolay değil.
- **SHAP arka planı:** Saldırı 10 kümeli arka planla kurulmuş olsa bile, test sırasında 5 ya da 15 küme kullanmak saldırıyı zayıflatmıyor. Arka planı tamamen sıfırlardan oluşturmak ise saldırıyı daha da belirgin hâle getiriyor.

### 7.5 Örnek açıklamalar

COMPAS'tan tek bir örnek için LIME katsayıları (makalenin Tablo 2'si) etkiyi çarpıcı biçimde gösterir. Saldırıdan önce ırkın katsayısı −0.42, ilişkisiz özelliğinki 0'dır. Saldırıdan sonra tablo tamamen yer değiştirir: ırk 0, ilişkisiz özellik 0.42. Katsayının büyüklüğü neredeyse aynen yeni özelliğe aktarılmıştır. SHAP'ta (makalenin Şekil 6'sı) ırkın katkısı belirgin biçimde küçülür ama LIME'daki gibi tamamen sıfırlanmaz. Bu da Bölüm 7.2'deki direnç açıklamasıyla uyumludur.

## 8. Makalenin konumu, sınırları ve sonrası

### 8.1 Literatürdeki yeri

Makale birkaç araştırma çizgisinin kesişiminde duruyor:

- **Sonradan açıklamalara yönelik eleştiriler:** Rudin, yüksek riskli kararlarda kara kutuyu açıklamaya çalışmak yerine baştan yorumlanabilir model kullanılması gerektiğini savunur. Ghorbani ve ark. açıklamaların küçük girdi değişikliklerine aşırı duyarlı olabildiğini gösterir. Mittelstadt ve ark. ise LIME/SHAP'ın yapay noktalarının tablo verisinde sezgisel olmadığına dikkat çeker. Bu makale son gözlemi alıp bir saldırıya dönüştürür.
- **Açıklamalara saldırılar:** Önceki çalışmalar (Dombrowski ve ark., Heo ve ark.) görüntü sınıflandırmada saliency haritalarının manipüle edilebildiğini göstermişti. Bu makale saldırıyı tablo verisine ve modelden bağımsız yöntemlere taşır.
- **Yorumlanabilirlik ve adalet:** Aïvodji ve ark.'nın "fairwashing" kavramı, yani açıklamaların haksız kararları meşrulaştırmak için kullanılması, bu makalenin motivasyonuyla doğrudan ilişkilidir.

### 8.2 Yazarların vardığı sonuç

Bozmaya dayalı sonradan açıklama yöntemleri, hassas uygulamalarda ayrımcılığı tespit etmek için tek başına yeterli değildir. LIME, SHAP'tan daha savunmasızdır. Yazarlar iki gelecek yön önerir: gradyan tabanlı yöntemlerin benzer saldırılara açık olup olmadığını incelemek ve bu tür saldırılara dayanıklı açıklama yöntemleri geliştirmek.

### 8.3 Eleştirel değerlendirme (kendi yorumum)

- **Güçlü varsayımlar:** Saldırgan, denetçinin hangi yöntemi ve hangi örnekleme ayarlarını kullanacağını biliyor. Yöntemler arası aktarımın zayıf olması, bu bilginin saldırı için kritik olduğunu gösteriyor.
- **Uç modeller:** *f* yalnızca tek bir özelliğe bakıyor. Gerçek önyargılar daha dağınıktır; bu durumda saldırının nasıl davranacağı ayrıca incelenmeli.
- **Denetçinin başka araçları:** Denetçi modeli yalnızca yapay noktalarda değil, gerçek veri noktalarında da sorgularsa (örneğin gerçek bir müşterinin yalnızca cinsiyetini değiştirip kararın değişip değişmediğine bakarsa) önyargı ortaya çıkabilir. Karşı olgusal açıklamaların ve basit "eşleştirilmiş test" yaklaşımının değeri buradan gelir. Ancak aynı ekibin 2021 tarihli *Counterfactual Explanations Can Be Manipulated* çalışması, karşı olgusal açıklamaların da manipüle edilebildiğini gösterir.
- **Savunma yönleri:** Veri dağılımına sadık örnekleme (özellikler arası ilişkiyi koruyan ya da gerçek veriden yararlanan örnekleme), açıklamaları farklı yöntemlerle çapraz kontrol etmek ve modelin dağılım dışı girdilerde gerçek veridekinden belirgin biçimde farklı davranıp davranmadığını test etmek.

## 9. Kendi yeniden üretimimde gözlenenler

Saldırının LIME kısmını, orijinal kodun mantığını izleyerek kendi kodumla yeniden uyguladım (`analiz.ipynb`; veri: `veri.csv`). İki veri üzerinde çalıştım: makalenin kullandığı önişlenmiş German Credit verisi ve uygulamamda kullandığım Kaggle German Credit sürümü. Kaggle sürümünde makalenin *ψ* özelliği (taksit/gelir oranı) bulunmadığı için *ψ* olarak kredi süresini (`Duration`, cinsiyetle korelasyonu 0.08) kullandım.

| Ölçüm                                      | Makale (German) | Orijinal veri, 5 bölme (ort. ± std) | Kendi verim (Kaggle) |
|--------------------------------------------|-----------------|-------------------------------------|----------------------|
| Taraflı modelde hassas özellik 1. sırada   | ~%100           | %100 ± 0                            | %100                 |
| Saldırgan modelde hassas özellik 1. sırada | ~%0             | %0 ± 0                              | %0 (ilk 3'te %1)     |
| Saldırgan modelde ψ'nin özelliği 1. sırada | %91             | %95.4 ± 6.4                         | %97                  |
| Sadakat                                    | %100            | %100 ± 0                            | %99                  |

- **Ana bulgu yeniden üretildi.** Bölmelere göre ψ oranı %84 ile %99 arasında değişti; makalenin tek bölmeyle raporladığı %91 bu aralığın içinde.
- **Dedektör eşiği:** Kendi verimde F1 ≤ 0.75 iken saldırı başarısız, F1 ≥ 0.92 iken tamamen başarılı. Makaledeki keskin geçiş burada da görülüyor. Bu veride dedektörün kolay başarılı olmasının muhtemel nedeni, yaş ve meslek gibi tam sayı değerli özelliklerin gürültüyle bu yapısını kaybetmesi.
- **Çekirdek genişliği:** 0.15·√*M* ile 0.95·√*M* arasındaki tüm değerlerde hassas özellik 1. sırada hiç çıkmadı. Küçük bir fark: En dar çekirdekte ψ'nin özelliği %58'e düştü; gizleme başarılı kaldı ama sahte gerekçe daha az öne çıktı.
- **Karşı olgusal uzantı:** Saldırgan modelin "kötü" dediği 30 müşterinin 27'sinde en küçük karşı olgusal değişiklik cinsiyeti değiştirmekti ve bu noktaların hepsi gerçek veri sayıldı. Yani karşı olgusal sorgu, LIME'ın gizlediği önyargıyı ortaya çıkardı. Kalan 3 müşteride ise arama, dedektörün "yapay" saydığı küçük bir sayısal değişiklik buldu ve bu yanıltıcı bir karşı olgusal açıklama üretti. Bu sonuç, karşı olgusal açıklamaların da aynı zafiyete açık olabileceğini gösteriyor.

## 10. Terim sözlüğü

| İngilizce                  | Türkçe                   | Kısa açıklama                                                          |
|----------------------------|--------------------------|------------------------------------------------------------------------|
| Post hoc explanation       | Sonradan açıklama        | Eğitilmiş modeli değiştirmeden kararlarını açıklayan yöntemler         |
| Perturbation               | Bozma / pertürbasyon     | Girdiyi değiştirerek yapay örnek üretme                                |
| Out-of-distribution (OOD)  | Dağılım dışı             | Eğitim verisinin dağılımına uymayan örnek                              |
| Data manifold              | Veri manifoldu           | Gerçek verinin yer aldığı (genellikle düşük boyutlu) bölge             |
| Scaffolding                | İskele                   | Taraflı modeli saran ve girdiye göre f ile ψ arasında geçiş yapan yapı |
| Adversarial classifier     | Saldırgan sınıflandırıcı | İskele ile kurulan e modeli                                            |
| Innocuous model            | Masum model              | Açıklamada gösterilmek üzere seçilen ψ                                 |
| Sensitive attribute        | Hassas özellik           | Irk, cinsiyet gibi ayrımcılığa konu olabilecek özellik                 |
| Fidelity                   | Sadakat                  | Gerçek veride e ile f'nin aynı kararı verme oranı                      |
| Local accuracy             | Yerel doğruluk           | SHAP katkılarının toplamının tahmin farkına eşit olması                |
| Background distribution    | Arka plan dağılımı       | SHAP'ta "eksik" özelliklerin değerlerinin alındığı dağılım             |
| Kernel width               | Çekirdek genişliği       | LIME'da uzaklığa göre ağırlığın ne kadar hızlı azaldığını belirler     |
| Fairwashing                | Adalet aklama            | Haksız bir modeli açıklamalarla adil gösterme                          |
| Counterfactual explanation | Karşı olgusal açıklama   | Kararın değişmesi için girdide gereken en küçük değişiklik             |

---

Kaynak: D. Slack, S. Hilgard, E. Jia, S. Singh, H. Lakkaraju. "Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods." Proceedings of the 2020 AAAI/ACM Conference on AI, Ethics, and Society (AIES '20). ACM. doi:10.1145/3375627.3375830 · Orijinal kod: github.com/dylan-slack/Fooling-LIME-SHAP

Diğer anılan çalışmalar: Ribeiro ve ark. 2016 (LIME); Lundberg ve Lee 2017 (SHAP); Slack ve ark. 2021, "Counterfactual Explanations Can Be Manipulated" (NeurIPS).
