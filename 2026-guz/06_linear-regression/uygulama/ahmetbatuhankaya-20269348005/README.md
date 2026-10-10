# Bölüm 6 – Doğrusal Regresyon: Uygulama

**Öğrenci:** Ahmet Batuhan Kaya (20269348005)
**Kaynak bölüm:** Molnar, *Interpretable Machine Learning*, Bölüm 6 – Linear Regression

## Amaç

Bölümde anlatılan doğrusal regresyon yorum araçlarını (katsayı yorumu, ağırlık grafiği, etki grafiği, tekil tahmin açıklaması, Lasso) **zorlu bir gerçek dünya veri setinde** uygulamak ve yöntemin sınırlarını göstermek.

## Veri setleri

| Veri | Kaynak | Kullanım |
|---|---|---|
| Bisiklet kiralama (kitabın örneği) | [Molnar'ın GitHub reposu](https://github.com/christophM/interpretable-ml-book) | Kitaptaki modelin Python'da yeniden üretimi |
| Ames Housing (2.930 ev) | De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data*. Journal of Statistics Education, 19(3). | Asıl uygulama |

Ames verisi neden zorlu: sağa çarpık hedef değişken, güçlü biçimde korelasyonlu öznitelikler (yaşam alanı–oda sayısı, garaj kapasitesi–garaj alanı) ve sıralı kategorik değişkenler.

## İçerik

| Adım | Ne yapılıyor | Kitapla bağlantısı |
|---|---|---|
| 1 | Kitaptaki bisiklet modeli `statsmodels` ile yeniden kuruluyor | Bölüm 6 örneği |
| 2 | Ames verisi hazırlanıyor, korelasyonlar inceleniyor | Çoklu bağlantı varsayımı |
| 3 | Ham fiyat ve log(fiyat) modelleri karşılaştırılıyor; artık grafikleri, Q-Q grafikleri, Breusch-Pagan testi | Eş varyanslılık ve normallik varsayımları |
| 4 | Katsayı tablosu, ağırlık grafiği, etki grafiği, tekil tahmin açıklaması | Yorumlama, ağırlık ve etki grafikleri |
| 5 | Doğrusal modelde SHAP değerinin kapalı formülü sayısal olarak doğrulanıyor | Etki kavramının merkezlenmiş hâli |
| 6 | VIF ve bootstrap ile çoklu bağlantının katsayılara etkisi | Sınırlılıklar: "oda sayısı negatif ağırlık alabilir" |
| 7 | Lasso düzenlileştirme yolu ve bootstrap ile seçim kararsızlığı | Seyrek doğrusal modeller |

## Temel bulgular

- **Varsayım ihlali:** Ham fiyatla kurulan model değişen varyans gösteriyor (Breusch-Pagan: 446) ve bir ev için negatif fiyat tahmin ediyor. log dönüşümü sorunu büyük ölçüde azaltıyor (Breusch-Pagan: 131); yorum çarpımsal hâle geliyor.
- **Yorum:** Diğer her şey sabitken genel kalite puanındaki 1 puanlık artış fiyatı yaklaşık %9,6 artırıyor. Oda sayısının katsayısı negatif: yaşam alanı sabitken oda eklemek, evi daha küçük odalara bölmek anlamına geliyor.
- **SHAP köprüsü:** φⱼ = βⱼ(xⱼ − E[xⱼ]) formülüyle hesaplanan katkıların toplamı, tahmin ile ortalama tahmin arasındaki farka birebir eşit.
- **Çoklu bağlantı:** 150 evlik bootstrap örneklerinde oda sayısı katsayısı %72, garaj katsayıları yaklaşık %24 oranında negatif çıkıyor. Tahmin kalitesi korunurken yorum çöküyor.
- **Lasso:** Kitaptaki gibi 5 öznitelikli modelde, garaj çiftinden yalnızca birinin seçildiği örnek oranı %54. Lasso'nun bir değişkeni seçmesi, o değişkenin önemli olduğu anlamına gelmiyor.

## Çalıştırma

Gerekli paketler:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn
```

Test edilen sürümler: Python 3.12, numpy 2.4, pandas 3.0, matplotlib 3.10, statsmodels 0.15, scikit-learn 1.8.

Çalıştırmak için:

```bash
python main.py
```

Ya da VS Code'da dosyayı açıp her `# %%` hücresinin üstündeki **Run Cell** ile adım adım çalıştırın. Veriler internetten otomatik indirilir. Grafikler `sekiller/` klasörüne kaydedilir. Tüm rastgelelik sabit tohuma bağlıdır (`seed=42`), dolayısıyla sonuçlar tekrarlanabilir.

## Kaynaklar

- Molnar, C. *Interpretable Machine Learning*. https://christophm.github.io/interpretable-ml-book/
- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417–473.
