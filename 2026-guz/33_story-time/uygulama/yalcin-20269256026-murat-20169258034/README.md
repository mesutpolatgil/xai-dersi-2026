# Bölüm 33–34 ortak uygulaması: Kararı açıkla, kaydı düzelt

**Yalçın Kavaklı — 20269256026 · Murat Şahin — 20169258034**

[İnteraktif uygulama HTML’si](uygulama.html) · [Jupyter / Colab defteri](uygulama.ipynb)

HTML dosyasını indirip tarayıcıda açın. Dört adım: yalnızca karar, gerekçeyi açma, vade kaydını düzeltme ve eşik karşılaştırması. Konuşmacı notları HTML içinden açılabilir.

33. bölümdeki Trust Fall hikâyesiyle bağlantılı olarak, bir otomatik kararın açıklanması ve hatalı kayıt üzerinden itiraz edilmesi canlandırılır. 34. bölümün amaçlar ve kısıtlar tartışması, karar eşiği ve hata türleri üzerinden ele alınır. Küçük karar ağacı bu süreci görünür kılan yardımcı araçtır.

Kaynak kaydın 9 aylık vadesi canlandırmada 48 ay olarak girilir; bu hata veri kümesinde raporlanmış bir olay değildir. Model gerçek verinin üç kavramıyla eğitilir; hedefi ele veren yapay bir sütun kullanılmaz.

Çalıştırma: `pip install -r requirements.txt` ardından `python main.py`.
Colab: `uygulama.ipynb` dosyasını yükle ve hücreleri sırayla çalıştır.

Veri: [UCI South German Credit](https://archive.ics.uci.edu/dataset/573/south%2Bgerman%2Bcredit%2B), Ulrike Grömping, 2020. Kaynak veri için UCI sayfasındaki CC BY 4.0 lisansı ve kaynak kodlama belgeleri geçerlidir. DOI: https://doi.org/10.24432/C5QG88
Kaynak ZIP adresi ve veri dosyası SHA-256 özeti sonuclar.json içinde kayıtlıdır.
Veri 1973–1975 döneminden; olumsuz krediler fazla örneklenmiştir. Ham tutar değerleri dönüştürülmüştür; güncel para birimi gibi sunulmaz.

Yöntem: üç gerçek kavram, kategoriler için ayrı göstergeler, max_depth=3, min_samples_leaf=30, random_state=42, sınıf oranları korunmuş %70/%30 ayrımı. Test: 210 olumlu, 90 olumsuz kayıt. %50 eşikte 176 doğru olumlu / 34 yanlış alarm / 54 kaçırılan olumsuz / 36 doğru olumsuz.

Bu bir kredi ön değerlendirme canlandırmasıdır. Bir kişinin gerçek uygunluğu hakkında karar vermez. Açıklama, veri kalitesi ve insan incelemesi birbirinden ayrı adımlardır. 

Kod ve özgün arayüz MIT lisanslıdır; veri lisansı ayrıdır. Python paketlerinin farklı sürümleri sonuçları değiştirebilir. HTML mevcut sonuclar.json içindeki ağacı kullanır; Python’da modeli tekrar eğitmek HTML’yi otomatik güncellemez.


Doğrulama: Python kodu ve notebook’un altı kod hücresi yerel Python’da çalıştırıldı. HTML hesap motorunun 1.000 gerçek satırdaki skor ve yaprak sonuçları Python ile karşılaştırıldı. Yeni defter için Colab arayüzü provası ayrıca yapılmalıdır.
