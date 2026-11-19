# Bölüm 22: Fonksiyonel Ayrıştırma (Functional Decomposition) Uygulaması

**Hazırlayan:** Yusuf Kaan Saçlı (20269258011)

## Uygulamanın Amacı
Bu uygulama, Christoph Molnar'ın *Interpretable Machine Learning* kitabının **22. Bölümü (Fonksiyonel Ayrıştırma / Functional Decomposition)** kapsamında hazırlanmıştır. Uygulamada kitaptaki sentetik tahmin fonksiyonu $f(x_1, x_2) = 2 + e^{x_1} - x_2 + x_1 \cdot x_2$ ($x_1, x_2 \in [-1, 1]$) üzerinden yüksek boyutlu tahmin yüzeylerinin anlaşılır alt bileşenlerine ayrıştırılması incelenmektedir:

1. **Bölüm 1 (Kitaptaki Şekil 3 ve Şekil 4):** Fonksiyonun 2 boyutlu tahmin yüzeyi (**Şekil 3**) ile sıfır ortalama (Fonksiyonel ANOVA) kuralına göre ayrıştırılmış ana etki bileşenleri $f_1(x_1)$, $f_2(x_2)$ ve ikili etkileşim yüzeyi (Pringles yüzeyi) $f_{12}(x_1, x_2)$ (**Şekil 4**) çizdirilerek `.png` olarak kaydedilir.
2. **Bölüm 2 (PyALE Kütüphanesi ile Ayrıştırma):** `PyALE` kütüphanesi (`ale`) kullanılarak model üzerinden 1 boyutlu ALE ana etki grafikleri (`x1` ve `x2`) ve 2 boyutlu ALE etkileşim yüzeyi (`['x1', 'x2']`) doğrudan kütüphane fonksiyonlarıyla hesaplanıp görselleştirilir.
3. **Bölüm 3 (GAM / GA2M Modeli):** `interpret` kütüphanesindeki `ExplainableBoostingRegressor` (EBM) kullanılarak aşağıdan yukarıya (bottom-up) Genelleştirilmiş Toplamsal Model (GAM / GA2M) eğitilir ve öğrenilen ana etki ile etkileşim terimlerinin önem skorları yazdırılır.

## Kurulum
Gerekli Python kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

## Çalıştırma
Uygulamayı doğrudan Python betiği olarak veya Jupyter Notebook üzerinden çalıştırabilirsiniz:

```bash
python main.py
```
veya
```bash
jupyter notebook bolum22_uygulama.ipynb
```
