# Bölüm 23 Uygulaması: Permütasyon Öznitelik Önemi (PFI)

**Öğrenci:** Yunus Emre Yazarlı (`20269258002`)  
**Kitap:** Christoph Molnar – *Interpretable Machine Learning* (23. Bölüm: *Permutation Feature Importance*)

## İçerik ve Amaç

Bu klasördeki `main.py` betiği, kitaptaki 23. bölüm analizlerini ana dizindeki `data/bike.csv` ve `data/penguins.csv` veri setlerini kullanarak **2/3 eğitim, 1/3 test** ayrımıyla sıfırdan çalıştırır ve bölümdeki 4 grafiği üretir:

1. **Bisiklet Kiralama Tahmini (SVM + PFI):**
   - `bike.csv` verisi üzerinde Destek Vektör Makinesi (`SVR`, RBF çekirdeği) eğitilir.
   - Test verisi üzerinde 50 tekrarlı permütasyon ile Ortalama Mutlak Hata (MAE) oranı hesaplanarak ortalama önem ve %5–%95 kantil aralıkları görselleştirilir (`fig-importance-bike-1.png`).
2. **Penguen Cinsiyet Sınıflandırması (3 Ayrı Lojistik Regresyon + PFI):**
   - `penguins.csv` verisinde her tür (`Adelie`, `Chinstrap`, `Gentoo`) için ayrı bir lojistik regresyon modeli eğitilir ve tüm yapı `species` değişkenine göre yönlendirme yapan tek bir kara kutu model olarak ele alınır.
   - Test verisi üzerinde log kaybı (`log_loss`) oranıyla PFI hesaplanarak görselleştirilir (`fig-importance-penguins-1.png`).
3. **Türe Göre Korelasyon Analizi:**
   - Yüzgeç uzunluğu (`flipper_length_mm`) ve vücut kütlesi (`body_mass_g`) arasındaki Pearson korelasyonu tüm veri (`all`) ve her tür alt kümesi için ayrı ayrı hesaplanıp çizilir (`fig-cor-by-species-1.png`).
4. **Grup Bazında (Türe Koşullu) PFI:**
   - Test verisi türlere göre alt gruplara ayrılarak her tür içinde bağımsız permütasyon uygulanır ve grup bazında PFI değerleri görselleştirilir (`fig-importance-penguins-by-species-1.png`).

## Gereksinimler

Python 3.9+ ve aşağıdaki kütüphaneler gereklidir:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Çalıştırma

Betik hem kendi klasöründen hem de proje kök dizininden doğrudan çalıştırılabilir (veri ve çıktı yollarını kendi konumuna göre otomatik çözer):

```bash
cd 23_feature-importance/uygulama/yunusemreyazarli-20269258002
python main.py
```

Çalıştırıldığında üretilen grafikler otomatik olarak aşağıdaki dizinlere kaydedilir:
- `23_feature-importance/yunusemreyazarli-20269258002_files/figure-html/`
- `23_feature-importance/feature-importance/figure-html/`

Ardından Quarto HTML çıktısını güncellemek için `23_feature-importance` klasöründe şu komut çalıştırılır:

```bash
cd ../..
quarto render yunusemreyazarli-20269258002.qmd --to html
```
