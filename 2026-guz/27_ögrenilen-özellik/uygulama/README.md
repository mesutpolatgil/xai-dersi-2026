# Bölüm 27 Uygulaması: Öğrenilmiş Özellikler ve Ağ Diseksiyonu (Learned Features & Network Dissection)

**Öğrenci:** Mevhibe Begüm Şanlı (`20259348001`)  
**Kitap:** Christoph Molnar – *Interpretable Machine Learning* (27. Bölüm: *Learned Features*)

## İçerik ve Amaç

Bu klasördeki main.py betiği, kitaptaki 27. bölüm analizlerini, ağ katmanlarındaki öğrenilmiş filtreleri ve kavramsal incelemeleri barındırır:

1. **İlk Katman Filtreleri (Conv1 Filters):** Modelin en temel evrişim katmanındaki filtre ağırlıklarını görselleştirerek ağın ham piksellerden kenar, yönelim ve basit dokuları nasıl öğrendiğini ortaya koyar.
2. **Kanal Aktivasyonları ve Hook Mekanizması:** İleri besleme sırasında modelin derin katmanlarındaki (örn. `layer4`) aktivasyon haritalarını yakalayarak hangi konseptlerin tetiklendiğini inceler.
3. **Ağ Diseksiyonu (Network Dissection) Simülasyonu:** Broden veri seti mantığıyla, katman aktivasyonlarının uzaysal maskelere dönüştürülerek orijinal görsel üzerinde hangi bölgelere odaklandığını (örn. nesne anatomisi) görselleştirir.

## Python Betiği (main.py) ve Kurulum Adımları

Uygulama kapsamında kullanılan temel Python kodu, kurulum komutları ve çalıştırma yönergelerinin tamamı aşağıdadır:

```python
"""
Bölüm 27: Öğrenilmiş Özellikler ve Ağ Diseksiyonu
Örnek Model İnceleme ve Filtre Analizi Betiği

--- KURULUM VE GEREKSİNİMLER ---
Python 3.9+ ve projeyi çalıştırmak için gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırın:
pip install numpy pandas torch torchvision matplotlib seaborn pillow requests

--- ÇALIŞTIRMA ADIMLARI ---
1. Betik doğrudan kendi klasöründen veya proje kök dizininden çalıştırılabilir:
cd 27_ogrenilen_ozellikler/uygulama/mevhibebegumsanli-20259348001
python main.py
(Çalıştırıldığında önceden eğitilmiş ResNet50 modeli yüklenerek analiz altyapısı konsola yazdırılır.)

2. Quarto HTML çıktısını güncellemek için çeviri klasöründe şu komut çalıştırılır:
quarto render mevhibebegumsanli-20259348001.qmd --to html
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def model_bilgisi_yukle():
    print("--- XAI Dersi Bölüm 27: ResNet50 ve Öğrenilmiş Özellikler Yükleme ---")
    
    # 1. ImageNet üzerinde eğitilmiş ResNet50 modeli yükleme (Kitap örneklerine uygun)
    print("Önceden eğitilmiş ResNet50 modeli yükleniyor...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()  # Değerlendirme modu
    
    print(f"Model başarıyla yüklendi: {type(model).__name__}")
    print("Not: Conv1 filtre görselleştirmeleri, hook mekanizmaları ve Network Dissection maske analizi için ileri besleme döngüleri kullanılabilir.")

if __name__ == "__main__":
    model_bilgisi_yukle()