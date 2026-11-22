# Bölüm 21: Özellik Etkileşimi (Feature Interaction) Uygulaması

**Hazırlayan:** Yusuf Kaan Saçlı (20269258011)

## Uygulamanın Amacı
Bu uygulama, Christoph Molnar'ın *Interpretable Machine Learning* kitabının **21. Bölümü (Özellik Etkileşimi / Feature Interaction)** kapsamında hazırlanmıştır. Uygulamada **Palmer Penguins** (`penguins.csv`) veri seti kullanılarak penguen cinsiyetini tahmin eden bir **Rastgele Orman (Random Forest)** modeli eğitilmekte ve özellikler arasındaki etkileşimler incelenmektedir:

1. **Bölüm 1 (PiML Kütüphanesi ile Etkileşim Analizi):** `piml` kütüphanesi (`Experiment`) kullanılarak vücut kütlesi (`body_mass_g`) özelliğinin diğer özelliklerle olan **Friedman H-İstatistiği (`hstat`)** etkileşim güçleri ve `body_mass_g` ile `bill_depth_mm` arasındaki **2 Boyutlu Kısmi Bağımlılık Grafiği (2B PDP)** doğrudan kütüphane fonksiyonlarıyla çizdirilir.
2. **Bölüm 2 (Kitaptaki Şekil 1 ve Şekil 2'nin Birebir Çizimi):** `scikit-learn` kütüphanesinin `partial_dependence` fonksiyonu kullanılarak `body_mass_g` ile diğer özelliklerin ikili H-istatistiği lolipop grafiği (**Şekil 1**) ve penguen türlerine (`Adelie`, `Chinstrap`, `Gentoo`) göre etkileşim gücü nokta grafiği (**Şekil 2**) oluşturulup `.png` olarak kaydedilir.

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
jupyter notebook bolum21_uygulama.ipynb
```
