# Bölüm 24 Uygulaması: Bir Özniteliği Dışarıda Bırakma (LOFO) Önemi

**Öğrenci:** Yunus Emre Yazarlı (`20269258002`)  
**Kitap:** Christoph Molnar – *Interpretable Machine Learning* (24. Bölüm: *Leave One Feature Out (LOFO) Importance*)

## İçerik ve Amaç

Bu klasördeki `main.py` betiği, kitaptaki 24. bölüm analizlerini ana dizindeki `data/bike.csv` ve `data/penguins.csv` veri setlerini kullanarak **2/3 eğitim, 1/3 test** ayrımıyla sıfırdan çalıştırır ve bölümdeki 3 LOFO grafiğini üretir:

1. **Bisiklet Kiralama Verisinde LOFO Önemi (`fig-lofo-bike-1.png`):**
   - `bike.csv` verisi üzerinde tüm özniteliklerle temel `RandomForestRegressor` modeli eğitilir ve test MAE değeri ölçülür.
   - Her bir öznitelik tek tek eğitim ve test setinden çıkarılıp model yeniden eğitilerek MAE farkı (`new_mae - base_mae`) hesaplanır ve görselleştirilir.
2. **Tam Korelasyonlu Öznitelik (`temp_copy`) ile LOFO Önemi (`fig-lofo-bike-correlated-1.png`):**
   - `temp` sütunu birebir kopyalanarak `temp_copy` (%100 korelasyon) veri setine eklenir.
   - Aynı eğitim/test ayrımıyla tüm öznitelikler için LOFO hesaplanarak korelasyonlu özniteliklerin koşullu önemlerinin sıfıra yaklaştığı gösterilir.
3. **Penguen Cinsiyet Sınıflandırmasında LOFO Önemi (`fig-lofo-penguins-1.png`):**
   - `penguins.csv` verisi üzerinde `RandomForestClassifier` eğitilir ve her öznitelik çıkarılarak yeniden eğitim yapılır.
   - Doğruluk (`accuracy`) ölçütünde büyük değer daha iyi olduğundan önem değeri `base_acc - new_acc` olarak hesaplanıp görselleştirilir.

## Gereksinimler

Python 3.9+ ve aşağıdaki kütüphaneler gereklidir:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Çalıştırma

Betik hem kendi klasöründen hem de proje kök dizininden doğrudan çalıştırılabilir (veri ve çıktı yollarını kendi konumuna göre otomatik çözer):

```bash
cd 24_lofo/uygulama/yunusemreyazarli-20269258002
python main.py
```

Çalıştırıldığında üretilen grafikler otomatik olarak aşağıdaki dizinlere kaydedilir:
- `24_lofo/yunusemreyazarli-20269258002_files/figure-html/`
- `24_lofo/lofo/figure-html/`

Ardından Quarto HTML çıktısını güncellemek için `24_lofo` klasöründe şu komut çalıştırılır:

```bash
cd ../..
quarto render yunusemreyazarli-20269258002.qmd --to html
```
