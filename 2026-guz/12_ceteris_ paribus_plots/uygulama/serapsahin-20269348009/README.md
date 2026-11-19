# Bölüm 12 – Ceteris Paribus Grafikleri

**Öğrenci:** Serap Şahin (20269348009)  
**Kaynak:** Christoph Molnar, *Interpretable Machine Learning*. [Bölüm 12 – Ceteris Paribus Plots](https://christophm.github.io/interpretable-ml-book/ceteris-paribus.html)

## Amaç

Tek bir gözlemde diğer girdiler sabitken bir özniteliği değiştirip modelin tahminini yeniden hesaplıyorum. Sayısal ve kategorik Ceteris Paribus (CP) profillerini Python’da yazıyorum. Ardından birbirine bağlı özniteliklerin bu açıklamayı nasıl etkilediğini inceliyorum.

## Veri setleri

| Veri seti | Uygulamadaki kullanımı | Kaynak |
| --- | --- | --- |
| Palmer Penguins | Gaga derinliği için tek gözlemli CP ve öznitelik bağımlılığı örneği | [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) |
| Bisiklet kiralama | Tek gün için what-if panelleri ve dört modelin sıcaklık profili | [Kitabın veri dosyası](https://github.com/christophM/interpretable-ml-book/blob/master/data/bike.csv) |
| IBM Telco Customer Churn | 7.043 müşteriden seçilen bir müşterinin ayrılma tahmini ve CP’nin sınırlılıkları | [IBM veri seti](https://github.com/IBM/telco-customer-churn-on-icp4d) |

## Uygulama akışı

| Adım | Yapılan çalışma |
| --- | --- |
| 1 | CP algoritmasını NumPy ve pandas ile kur. Sayısal ve kategorik sonuçları, her senaryo için modeli ayrı çağıran döngüyle doğrula. |
| 2 | Kitaptaki 20,6 mm gaga derinliğine sahip Adelie penguenini ve 3 Ocak 2011 bisiklet gözlemini eğitim dışında tutarak açıkla. |
| 3 | Bisiklet örneğinde SVM what-if panellerini çiz. Sıcaklık profillerini doğrusal model, karar ağacı, rastgele orman ve SVM için karşılaştır. |
| 4 | Telco’da sayısal ve kategorik CP profilleri üret. Abonelik süresi, toplam ödeme ve internet hizmeti arasındaki bağımlılıkları incele. |
| 5 | Klasik CP’yi, ilişkili girdilerin birlikte güncellendiği senaryoyla karşılaştır. Model yeniden eğitildiğinde eğrinin ne kadar değiştiğini bootstrap örneklemleriyle ölç. |

Türkçe çeviride Şekil 12.2, dişi sınıfı açıkça seçilerek yeniden çizilmiştir. Python uygulaması da aynı 20,6 mm’lik Adelie pengueni için `P(dişi)` olasılığını gösterir. Özgün kitaptaki şekilde ise eksen dişi olarak etiketlense de çizilen olasılık erkek sınıfına aittir. R ve Python’daki eğitim ayrımı ve model uygulaması, eğrinin sayısal ayrıntılarını değiştirebilir.

## Temel bulgular

- **Doğrulama.** Altı gaga derinliği ve üç sözleşme türü için CP sonuçları referans döngüsüyle eşleşti.
- **Tek müşteri için model tahmini.** `5949-XIKAE` müşterisinin aylık sözleşmesinde tahmin 0,62. Sözleşme girdisi iki yıllık olarak değiştirildiğinde model tahmini 0,07 oluyor. Bu fark, sabit modelin senaryoya verdiği tepkiyi gösteriyor.
- **Sayısal bağımlılık.** Abonelik süresi tek başına değiştirildiğinde CP noktalarının %90’ı en yakın komşu uzaklığı eşiğini aştı. Toplam ödeme de müşteri oranı korunarak güncellendiğinde bu oran %0 oldu. Eşik, eğitim verisindeki yakınlığı ölçüyor.
- **Kategorik kısıt.** İnternet hizmeti kapatılıp bağlı hizmetler sabit bırakıldığında oluşan kategori birleşimi eğitim verisinde görülmedi. Model tahmini bu senaryoda 0,32, bağlı hizmetler de güncellendiğinde 0,22 oldu.

## Çalıştırma

Python 3.13 ile uygulama klasöründe aşağıdaki komutları çalıştırın.

```bash
python -m pip install -r requirements.txt
python main.py
```

Veri dosyaları GitHub’dan indirildiği için internet bağlantısı gerekir. Grafikler Matplotlib pencerelerinde açılır. VS Code’da Python ve Jupyter uzantılarıyla `# %%` hücrelerini **Run Cell** üzerinden ayrı ayrı çalıştırabilirsiniz.

## Lisans

Kitap içeriği ve bu uygulama [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) lisansı altındadır.
