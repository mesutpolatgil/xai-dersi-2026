# xai-dersi-2026

Bu depo, Christoph Molnar tarafından yazılan 'Interpretable Machine Learning' kitabının Türkçe çevirisini ve ilgili Python laboratuvar uygulamalarını içermektedir. Orijinal eser [CC BY-NC-SA 4.0] lisansı altındadır ve bu çeviri de aynı lisansla, ticari olmayan eğitim amaçlarıyla açık kaynak olarak sunulmaktadır. Orijinal kitaba buradan ulaşabilirsiniz: https://christophm.github.io/interpretable-ml-book/

dosya yapısı

```
xai-dersi-2026/
  └── 2026-guz/
      │
      ├── 01_introduction/
      │   ├── ceviri/
      │   │   ├── ahmet-1234.md
      │   │   └── ayse-5678.md
      │   ├── sunum/
      │   │   ├── ahmet-1234.pdf
      │   │   └── ayse-5678.pdf
      │   ├── uygulama/
      │   │   ├── ahmet-1234/
      │   │   │   ├── main.py
      │   │   │   └── README.md
      │   │   └── ayse-5678/
      │   │       └── main.py
      │   └── makale-yeniden-uretim/
      │       ├── ahmet-1234/
      │       │   ├── makale.md
      │       │   ├── veri.csv
      │       │   └── analiz.ipynb
      │       └── ayse-5678/
      │           └── makale.md
      │
      ├── 02_interpretability/
      │   └── (aynı yapı)
      │
      ├── 03_goals-of-interpretability/
      │   └── (aynı yapı)
      │
      ├── ...
      │
      └── 34_the-future-of-interpretability/
          └── (aynı yapı)
```


---

## 📄 Render Kuralı (Zorunlu)

Her öğrenci `.qmd` dosyasını **kendi bilgisayarında** render etmek zorundadır:


Bu komut aynı klasöre şunları üretir:
- `dosya.html` → GitHub'da görüntülenebilir site
- `dosya_files/` → CSS, görsel, font klasörü

**PR'a hem `.qmd` hem `.html` hem `dosya_files/` klasörü eklenmelidir.**

Sadece `.qmd` gönderen PR reddedilir — çünkü GitHub `.qmd`'yi ham metin olarak gösterir.

**Quarto kurulumu:** https://quarto.org/docs/get-started/
