# Önem Haritaları İçin Sağlama Testleri

**Tania El Hawari · 20259258043**

Bu çalışma, *Sanity Checks for Saliency Maps* makalesinin iki rastgeleleştirme testini özgün sentetik veri ve küçük bir evrişim ağıyla yeniden uygular.

Ödev üç ayrı etkinlik içerir. Kuramsal anlatımda defterin matematik ve yöntem bölümleri kullanılır. **Temel Uygulama / Canlı Kod** için öğrencinin kendi çektiği gerçek fotoğraflar gerekir. **Makale Yeniden Üretimi** için 416 sentetik görüntüyle kaydedilen kontrollü Seçenek 1 deneyi kullanılır. Seçenek 1 kapsamında, model veya eğitim etiketleri değiştirildiğinde ikna edici görünen bir önem haritasının ne ölçüde değiştiğini araştırır. [Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt ve Been Kim, NeurIPS 2018, arXiv v3](https://arxiv.org/html/1810.03292v3).

## Dosyalar

| Dosya veya klasör | İçerik |
|---|---|
| `analiz.ipynb` | Veri üretimi, model eğitimi, yedi açıklama yöntemi, deneyler, ölçümler, grafikler ve rapor üretimi |
| `veri.csv` | Kimlik, dosya yolu, sınıf, bölme, koşullar, tohum, köken ve SHA-256 özeti |
| `data/synthetic/` | 416 özgün sentetik PNG, bölmelere göre ayrılmış |
| `data/raw/` | Öğrencinin sağlayacağı fotoğraflar ve manifesto |
| `sekiller/` | Defterin ürettiği araştırma görselleri |
| `foto-sonuclari/` | Fotoğraf modu çalıştırıldığında oluşan ayrı CSV, grafik ve raporlar |
| `makale.md`, `makale.html` | Yürütülen deneylerden üretilen Türkçe rapor ve tarayıcı çıktısı |
| `requirements.txt` | Kullanılan deney paketlerinin sürümleri |

## Kurulum ve yeniden çalıştırma

Aşağıdaki Windows PowerShell komutları proje klasöründe bir Python 3.13 sanal ortamı oluşturur ve gerekli paketleri kurar. Deney PyTorch’un işlemci sürümüyle yürütüldü. JupyterLab veya VS Code’un Jupyter uzantısı aynı sanal ortamla kullanılabilir.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyterlab
```

`analiz.ipynb` dosyası, yeni bir Python çekirdeğinde hücrelerin sırasıyla yürütülmesiyle yeniden üretilir. `PHOTO_MODE = False` iken veri hücresi aynı tohumdan PNG’leri ve `veri.csv` dosyasını üretir. Eğitim ve deney hücreleri `sekiller/` içindeki görselleri yeniler. Son hücre, koşunun sayıları ve mevcut Markdown açıklamalarından `makale.md` dosyasını oluşturur ve bu dosyayı `makale.html` olarak işler. Rapor, göreli görsel yollarını kullandığı için HTML ile `sekiller/` aynı dizin düzeninde bulunmalıdır. Matematik için MathJax çevrimiçi yüklenir.

## Kayıtlı çıktıları doğrulama

Proje klasöründe, bağımlılıkların kurulu olduğu sanal ortamla şu denetim çalıştırılabilir:

```powershell
.\.venv\Scripts\python.exe verify_project.py
```

Betik, sentetik veri manifestosunu ve görüntüleri, kaydedilmiş defter çıktılarını, rapor tablolarını ve göreli şekil yollarını kontrol eder. Eğitimi yeniden başlatmaz. Ham benzerlik kayıtları ayrı bir dosya olarak saklanmadığından, sayısal çapraz kontrol defterde yazdırılan ölçümlerle sınırlıdır. Fotoğraf modu ve tarayıcıdaki MathJax gösterimi bu denetimin kapsamı dışındadır.

## Veri kökeni ve fotoğraf seçeneği

Varsayılan veri, halka ve artı şekillerinin programla oluşturulmuş 32×32 gri tonlu görüntülerinden oluşur. Eğitim, doğrulama ve test bölmeleri sırasıyla 256, 64 ve 96 örnek içerir. Gürültü, örtülme, desen, yön ve konum değişir. Köşe işareti eğitimde sınıfla ilişkilidir. Doğrulama ve testte aynı ipucu sınıftan bağımsız örneklenir. Dış veri veya hazır model ağırlığı kullanılmaz.

### Hangi fotoğraflar gerekli?

- **0 · halka:** Yüzük, conta, halka oyuncak veya rondela gibi ortası boş nesneler.
- **1 · arti:** İki kolu ortada kesişen fiziksel nesneler. Artı biçimli oyuncak, parça veya kesilmiş karton bu sınıfa örnektir. Her fotoğrafta tek bir hedef sınıf bulunur.
- **Örnek sayısı:** Yükleyicinin kabul ettiği asgari sayı **48 gerçek fotoğraftır**. Eğitim, doğrulama ve test bölmelerinin her birinde her sınıftan en az sekiz görüntü gerekir. Bu sayı öğretmenin doğrulanmış kotası veya öğrenme başarısının güvencesi değildir. Daha kapsamlı bir pilot çalışma için sınıf başına yaklaşık 160 görüntü, gruplar elverdiğinde 112 eğitim, 24 doğrulama ve 24 test örneği önerilir.
- **Görüntü özellikleri:** JPEG veya PNG kullanılır. Kısa kenarın tercihen en az 640 piksel olması, yükleyicinin 64 piksel alt sınırının üzerinde veri sağlar. Nesnenin kadrajın yaklaşık üçte biri ile üçte ikisi arasında yer kaplaması, 32×32 gri tonlu model girdisinde görünürlüğünü destekler.
- **Görsel çeşitlilik:** İki sınıfta benzer arka plan ve ışık dağılımı hedeflenir. Düz ve desenli zeminler, farklı açılar ve mesafeler, gölge, kısmi örtülme ve benzer şekilli dikkat dağıtıcılar zorluğu artırır. Test bölmesi, eğitimde bulunmayan fiziksel nesne ve çekim oturumlarından oluşur.

### Manifesto ve veri sızıntısı kontrolü

Aynı fiziksel nesnenin ve ilişkili çekim oturumlarının bütün görüntüleri aynı `group_id` ile tek bölmeye atanır. Kırpılmış veya yeniden boyutlandırılmış görüntüler de kaynak görüntüyle aynı grupta kalır. Defter grup çakışmasını, aynı dosya içeriğini ve aynı 32×32 model girdisini denetler. Yakın kopyaların ayrılmasında çekim ve nesne bilgisi esas alınır.

Girdi dosyaları `data/raw/` altında yer alır. `data/raw/foto_manifest.csv` dosyasının kodlaması UTF-8’dir ve dosya yolları proje köküne göredir. Örnek şema şöyledir:

```csv
image_id,path,label,class_name,split,group_id
ornek-001,data/raw/ornek-001.jpg,0,halka,train,nesne-01
ornek-002,data/raw/ornek-002.jpg,1,arti,validation,nesne-02
```

Örnek satırlar dosya biçimini gösterir. Gerçek fotoğrafların manifestosu `train`, `validation` ve `test` bölmelerini içerir. Fotoğraf koşusu, `PHOTO_MODE = True` ayarıyla temiz bir çekirdekte yürütülür. Kişisel fotoğraflar, giriş manifestosu ve fotoğraf koşusunun çıktıları varsayılan Git dışlama kuralları kapsamındadır. Fotoğraf koşusunun manifestosu, grafikleri ve raporları `foto-sonuclari/` altında üretilir. Kök dizindeki sentetik CSV, görseller ve rapor bu moddan etkilenmez. Sentetik koşu, `PHOTO_MODE = False` ayarıyla temiz bir çekirdekte yürütülür.

Kaydedilmiş sentetik koşuda temel eğitim, üç tohumlu bağımsız ve kademeli parametre rastgeleleştirmesi ve tek etiket permütasyonu tamamlandı. Gerçek etiketli test doğrulukları sırasıyla %98,96 ve %43,75 oldu. Kişisel fotoğraf henüz sağlanmadığı için gerçek dünya eğitimi veya canlı demo doğrulanmadı.

### Canlı kod hazırlığı

VS Code’da defter, Python ve Jupyter uzantılarıyla `.venv` çekirdeğinde çalışır. İlk fotoğraf koşusunun değerlendirmesi, 32×32 önizlemeleri, sınıf dengesini, test doğruluğunu ve karışıklık matrisini kapsar. Canlı gösterimde 6. bölümde aynı girdi için Gradient ve Grad-CAM, 10. bölümde ise işaretli ve mutlak ölçekler karşılaştırılabilir. Başarısı sınırlı bir modelin açıklamaları, öğrenilmiş sınıf ilişkisinin kanıtı olarak yorumlanamaz.

## Yöntem, kapsam ve AI katkısı

Defter, girdi gradyanı, Gradient × Input, Integrated Gradients, SmoothGrad, Guided Backpropagation, Grad-CAM ve Guided Grad-CAM hesaplamalarını içerir. PyTorch katman, optimizasyon ve türev API’lerini sağlar. Açıklama formülleri, rastgeleleştirme ve benzerlik karşılaştırmaları bu proje için yazılmıştır. Hazır XAI paketi veya dış uygulama kodu kullanılmaz.

Çalışma makalenin temel müdahalelerini dener. Makaledeki büyük ağları, veri kümelerini veya sayısal sonuçları yeniden üretme iddiası taşımaz. Yalnızca bir eğitim tohumu ve bir etiket permütasyonu kullanılır. Parametre rastgeleleştirmesinde üç tohum aynı 16 görüntü üzerinde değerlendirilir. Sentetik veride gözlenen etkilerin doğal fotoğraflara aktarımı sınanmamıştır.

Yapay zekâ destekli kodlama yardımı defterin, rapor metinlerinin ve teknik doğrulama iş akışının hazırlanmasında kullanıldı.
