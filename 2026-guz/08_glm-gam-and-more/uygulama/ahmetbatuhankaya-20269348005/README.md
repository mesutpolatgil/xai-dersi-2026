# Bölüm 8 – GLM, GAM ve ötesi: Uygulama

**Öğrenci:** Ahmet Batuhan Kaya (20269348005)
**Kaynak bölüm:** Molnar, *Interpretable Machine Learning*, Bölüm 8 – GLM, GAM and more

## Amaç

Kitabın doğrusal modelin üç sorununa getirdiği çözümleri (GLM, etkileşim terimleri, GAM) önce kitabın kendi örnekleriyle yeniden üretmek, sonra **zorlu bir gerçek dünya veri setinde** uygulamak ve bu çözümlerin de kırılabildiği noktaları göstermek.

## Veri setleri

| Veri | Kaynak | Kullanım |
|---|---|---|
| Simüle kahve verisi | Kitaptaki kod | Doğrusal model ile Poisson GLM'in karşılaştırılması, güven aralığı kapsama deneyi |
| Bisiklet kiralama | Molnar'ın GitHub reposu | Etkileşim, doğrusal olmayan etki, log(y) ile log bağlantı karşılaştırması |
| freMTPL2freq (677.991 poliçe) | Fransız araç sigortası; CASdatasets / OpenML (id 41214) | Asıl uygulama |

freMTPL2 neden zorlu: poliçelerin %96'sında hasar yok, poliçeler farklı sürelerle gözlenmiş (offset gerekiyor), aşırı yayılım var, sürücü yaşının etkisi doğrusal değil ve Area ile Density neredeyse aynı bilgiyi taşıyor.

## İçerik

| Adım | Ne yapılıyor | Kitapla bağlantısı |
|---|---|---|
| 1 | Kahve simülasyonu: doğrusal model negatif tahmin üretiyor, Poisson GLM gerçek katsayıları geri kazanıyor; 200 tekrarla %95 güven aralığının kapsama oranı | GLM örneği |
| 2 | Bisiklet: sıcaklık × iş günü etkileşimi ve iki eğim | Etkileşimler |
| 3 | Bisiklet: sıcaklık için doğrusal, karekök, kategorize ve GAM modelleri | Doğrusal olmayan etkiler (kitaptaki 4 panelli şekil) |
| 4–5 | freMTPL2: doğrusal model vs offset'li Poisson GLM, çarpımsal yorum | GLM |
| 6 | Aşırı yayılım testi, quasi-Poisson ve Negatif Binom | GLM varsayımları |
| 7 | Sürücü yaşı: doğrusal, kategorize ve GAM (pyGAM) | GAM |
| 8 | log(y) modellemek ile log bağlantı fonksiyonunun farkı | Dönüşüm ve bağlantı fonksiyonu |
| 9 | Area ve Density: aynı bilgiyi taşıyan öznitelikler | Çoklu bağlantı / concurvity |

## Temel bulgular

- **Kahve:** Doğrusal model negatif kahve sayısı tahmin ediyor; Poisson GLM gerçek katsayılara yakın sonuç veriyor (stres 0,13 / gerçek 0,10; uyku −0,18 / gerçek −0,20). 200 tekrarın %95,5'inde %95 güven aralığı gerçek değeri kapsıyor.
- **Etkileşim:** Etkileşim katsayısı −31,0 olsa da iş günündeki sıcaklık eğimi 78,1 − 31,0 = 47,1, yani hâlâ pozitif.
- **Offset ve çarpımsal yorum:** BonusMalus 10 puan artınca hasar sıklığı 1,30 katına çıkıyor; benzinli araçlarda sıklık dizele göre 0,83 katı.
- **Aşırı yayılım:** Pearson χ²/sd = 1,66. Poisson modeli standart hataları yaklaşık %22 küçük hesaplıyor; Negatif Binom AIC'si daha düşük.
- **Sürücü yaşı:** Doğrusal GLM 18 yaşındaki sürücüyü 45 yaşındakinden *daha az* riskli (0,87 kat) gösterirken GAM 1,44 kat daha riskli buluyor. Ham verideki fark daha da büyük, çünkü genç sürücülerin BonusMalus puanı da yüksek.
- **log(y) ≠ log bağlantı:** log(cnt) modeli toplam kiralamayı 132.000 eksik tahmin ediyor; log bağlantılı Poisson GLM toplamı tam olarak koruyor.
- **Area–Density:** Spearman korelasyonu 0,98. Tek başına Area'da E bölgesi %55 daha riskli; Density eklenince Area katsayılarının hepsi anlamsızlaşıyor.

## Çalıştırma

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn pygam
python main.py
```

Ya da VS Code'da her `# %%` hücresini **Run Cell** ile çalıştırın. Sigorta verisi ilk çalıştırmada OpenML'den indirilir (yaklaşık 1 dakika) ve aynı klasöre `freMTPL2freq.csv` olarak kaydedilir; sonraki çalıştırmalar bu dosyayı kullanır. Grafikler `sekiller/` klasörüne kaydedilir; rastgelelik `seed=42` ile sabitlenmiştir. GAM, hız için 200.000 poliçelik rastgele bir örnek üzerinde eğitilir.

## Kaynaklar

- Molnar, C. *Interpretable Machine Learning*. https://christophm.github.io/interpretable-ml-book/
- Noll, A., Salzmann, R., & Wüthrich, M. V. (2020). Case Study: French Motor Third-Party Liability Claims. SSRN 3164764.
- Dutang, C., & Charpentier, A. CASdatasets: Insurance datasets (R paketi).
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2. baskı). CRC Press.
- Servén, D., & Brummitt, C. (2018). pyGAM: Generalized Additive Models in Python. Zenodo.
