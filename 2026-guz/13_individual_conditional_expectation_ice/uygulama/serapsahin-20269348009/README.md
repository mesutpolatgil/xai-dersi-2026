# Bölüm 13 – Bireysel Koşullu Beklenti (ICE)

**Öğrenci:** Serap Şahin (20269348009)  
**Kaynak:** Christoph Molnar, *Interpretable Machine Learning*. [Bölüm 13 – Individual Conditional Expectation](https://christophm.github.io/interpretable-ml-book/ice.html)

## Amaç

Her gözlem için bir özniteliğin değerini değiştirip modelin tahmin eğrisini hesaplıyorum. ICE, merkezlenmiş ICE (c-ICE) ve türev ICE (d-ICE) hesaplarını Python’da yazıyorum. California Housing örneğinde bireysel eğrileri PDP’nin ortalamasıyla karşılaştırıp öznitelik bağımlılığının yarattığı sınırlılıkları inceliyorum.

## Veri setleri

| Veri seti | Uygulamadaki kullanımı | Kaynak |
| --- | --- | --- |
| Bisiklet kiralama | Sıcaklık, nem ve rüzgâr hızı ICE’leri ile merkezlenmiş sıcaklık eğrileri | [Kitabın veri dosyası](https://github.com/christophM/interpretable-ml-book/blob/master/data/bike.csv) |
| Palmer Penguins | Gaga uzunluğu için sınıf etiketi ve olasılık ICE’leri | [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) |
| California Housing | 20.640 bölgede gelir ve bina yaşına göre model davranışı ile eğitim verisine uzak senaryolar | [California Housing veri dosyası](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) |

## Uygulama akışı

| Adım | Yapılan çalışma |
| --- | --- |
| 1 | ICE matrisini NumPy ve pandas ile oluştur. Aynı ızgarada scikit-learn çıktısıyla karşılaştır. |
| 2 | Kitaptaki bisiklet ve penguen grafiklerini Python’a uyarla. Mevsime göre renklendirilmiş nem eğrilerini, c-ICE ve d-ICE grafiklerini çiz. |
| 3 | California Housing modelinde medyan gelir için ICE ve PDP’yi karşılaştır. c-ICE eğrilerini bina yaşına göre incele ve kümele. |
| 4 | En yakın komşu uzaklığıyla eğitim verisine uzak ICE noktalarını işaretle. Enlemi tek başına değiştirmenin sonucunu harita üzerinde göster. |

## Temel bulgular

- **Doğrulama.** Bisiklet sıcaklık örneğinde ICE matrisi ile scikit-learn çıktısı arasındaki maksimum mutlak fark 0.
- **Model performansı.** California Housing modelinin test R² değeri 0,838, ortalama mutlak hatası yaklaşık 31 bin dolar.
- **PDP’nin gizlediği farklılık.** Alt referans noktasından orta gelir düzeyine kadar model tahminindeki ortalama artış eski bina grubunda 172 bin dolar, yeni bina grubunda 119 bin dolar. c-ICE bu farkı görünür kılıyor.
- **Veri desteği.** Seçilen uzaklık eşiklerini ICE ızgara noktalarının enlemde %65,5’i, toplam oda sayısında %79,8’i ve nüfusta %85,5’i aştı. Bu oranlar kullanılan ızgara ve uzaklık ölçütüne bağlı.

## Çalıştırma

Python 3.13 ile uygulama klasöründe aşağıdaki komutları çalıştırın.

```bash
python -m pip install -r requirements.txt
python main.py
```

Veri dosyaları GitHub’dan indirildiği için internet bağlantısı gerekir. Grafikler Matplotlib pencerelerinde açılır. VS Code’da Python ve Jupyter uzantılarıyla `# %%` hücrelerini **Run Cell** üzerinden ayrı ayrı çalıştırabilirsiniz.

## Lisans

Kitap içeriği ve bu uygulama [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) lisansı altındadır.
