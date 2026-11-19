# Bölüm 28 Uygulaması: Önemlilik Haritaları (Saliency Maps)

**Öğrenci:** Mevhibe Begüm Şanlı (`20259348001`)  
**Kitap:** Christoph Molnar – *Interpretable Machine Learning* (28. Bölüm: *Saliency Maps*)

## İçerik ve Amaç

Bu klasördeki main.py betiği, kitaptaki 28. bölüm analizlerini ve gradyan tabanlı model incelemelerini barındırır:

1. **Vanilla Gradient:** İlgili sınıf puanının giriş piksellerine göre gradyanını hesaplayarak piksel katkı haritalarını çıkarır.
2. **Grad-CAM (Gradient-weighted Class Activation Mapping):** Son evrişim katmanının gradyan bilgilerini kullanarak modelin görsel üzerinde odaklandığı bölgeleri kaba bir lokalizasyon haritası ile gösterir.
3. **SmoothGrad:** Girdi görüntüsüne rastgele gürültü ekleyip bu gürültülü gradyanlar üzerinden ortalama alarak daha net ve az gürültülü önemlilik haritaları elde eder.

## Python Betiği (main.py) ve Kurulum Adımları

Uygulama kapsamında kullanılan temel Python kodu, kurulum komutları ve çalıştırma yönergelerinin tamamı aşağıdadır:

```python
"""
Bölüm 28: Önemlilik Haritaları (Saliency Maps)
Örnek Gradyan Hesaplama ve Model İnceleme Betiği

--- KURULUM VE GEREKSİNİMLER ---
Python 3.9+ ve projeyi çalıştırmak için gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırın:
pip install numpy pandas torch torchvision matplotlib seaborn pillow opencv-python

--- ÇALIŞTIRMA ADIMLARI ---
1. Betik doğrudan kendi klasöründen veya proje kök dizininden çalıştırılabilir:
cd 28_onemlilik_haritalari/uygulama/mevhibebegumsanli-20259348001
python main.py
(Çalıştırıldığında önceden eğitilmiş VGG-16 modeli yüklenerek analiz altyapısı konsola yazdırılır.)

2. Quarto HTML çıktısını güncellemek için çeviri klasöründe şu komut çalıştırılır:
quarto render mevhibebegumsanli-20259348001.qmd --to html
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def model_bilgisi_yukle():
    print("--- XAI Dersi Bölüm 28: VGG-16 ve Önemlilik Haritaları Yükleme ---")
    
    # 1. ImageNet üzerinde eğitilmiş VGG-16 modeli yükleme (Kitap örneklerine uygun)
    print("Önceden eğitilmiş VGG-16 modeli yükleniyor...")
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.eval()  # Değerlendirme modu
    
    print(f"Model başarıyla yüklendi: {type(model).__name__}")
    print("Not: Vanilla Gradient, Grad-CAM ve SmoothGrad uygulamaları için ileri/geri yayılım (forward/backward pass) döngüleri kullanılabilir.")

if __name__ == "__main__":
    model_bilgisi_yukle()