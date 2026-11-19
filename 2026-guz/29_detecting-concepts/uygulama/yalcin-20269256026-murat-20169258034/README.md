# Bölüm 29 ortak uygulaması: Kavramların Tespiti (TCAV) ile Kısayol Denetimi ve Model İyileştirme

**Yalçın Kavaklı — 20269256026 · Murat Şahin — 20169258034**

[İnteraktif uygulama HTML’si](uygulama.html) · [Sunucu ve öğrenme rehberi HTML’si](uygulama_rehber.html) · [Jupyter / Colab defteri](uygulama.ipynb)

HTML dosyasını indirip tarayıcıda açın. Dört adım: (1) yüksek doğruluk yanılsaması ve kısayol tuzağı, (2) Kavram Aktivasyon Vektörü (CAV) ve yönlü türev, (3) küresel TCAV skoru ve iki yönlü t-testi (Bonferroni düzeltmesiyle), (4) veri yanlılığı temizlenerek eğitilen düzeltilmiş modelin TCAV karşılaştırması. Konuşmacı notları HTML içinden açılabilir.

29. bölümde (*Detecting Concepts*) anlatılan Kim vd. (2018) TCAV yöntemi, tekil piksel öneminin ötesine geçerek insan dostu soyut kavramların sinir ağının gizli katmanlarındaki (darboğaz / bottleneck) etkisini ölçer. Kitabın “Güçlü Yönler” bölümündeki zebra örneğine paralel olarak; eğitim kümesinde `"0"` sınıfı görüntülerinin %90'ına sağ üst 2×2 köşede küçük parlak bir damga (shortcut artifact) karıştığında, sinir ağının temiz test kümesinde **%96,9** genel doğruluk verse bile `"0"` sınıfı tahminlerinde **Köşe Damgası** kavramına **%100,0** ($p = 0,0045$) bağımlı hâle geldiği TCAV ile tespit edilir. Eğitim verisindeki damga yanlılığı temizlendiğinde **Halkalı Gövde (Ring)** TCAV skoru **%99,2** ($p = 0,0004$) ile korunurken, **Köşe Damgası** TCAV skoru **%19,0**'a düşer ve damgalı diğer rakamlarda yanılma oranı **%48,4'ten %0,0'a** iner. **Benekli Gürültü (Dotted Noise)** kontrol kavramı ise iki yönlü t-testini geçemeyerek ($p > 0,05$, `*`) elenir.

Çalıştırma: `pip install -r requirements.txt` ardından `python main.py`.
Colab: `uygulama.ipynb` dosyasını yükle ve hücreleri sırayla çalıştır.

Veri: `scikit-learn` içinde yer alan [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) (1.797 adet 8×8 piksel el yazısı rakam görüntüsü, 10 sınıf). %70 eğitim (1.257) / %30 test (540 görüntü, 54 adet `"0"` sınıfı) ayrımı, `random_state=42`.

Yöntem: İki gizli katmanlı sinir ağı (`MLPClassifier(hidden_layer_sizes=(48, 24), activation='relu', random_state=42)`), 3 kavram kümesi (her biri 45 örnek) ve $N=12$ rastgele veri kümesi. Her darboğaz katmanında ($l \in \{0, 1\}$) Lojistik Regresyon ile 12 CAV eğitilir; hedef sınıf ($k=0$) log-olasılık gradyanı ile yönlü türev $S_{C,0,l}(\mathbf{x}) = \nabla h_{l,0}(\hat{f}_l(\mathbf{x})) \cdot \mathbf{v}_l^C$, küresel TCAV oranı ve iki yönlü Welch t-testi hesaplanır.

Kod ve özgün arayüz MIT lisanslıdır. HTML, `sonuclar.json` içindeki eğitilmiş ağ ağırlıklarını ve CAV vektörlerini kullanarak 8×8 piksel ızgarasındaki her değişikliği tarayıcıda canlı hesaplar; internet bağlantısı gerektirmez.
