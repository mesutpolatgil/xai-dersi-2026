# Bölüm 30 Ek Uygulama: Fashion-MNIST ile Kötü Amaçlı Örnekler

Christoph Molnar'ın *Interpretable Machine Learning* kitabının 30. bölümündeki saldırı yöntemlerinin Fashion-MNIST veri seti üzerinde uygulanması.

- **Veri seti:** Fashion-MNIST, 28×28 piksellik gri tonlamalı giysi görüntüleri, 10 kategori (Tişört, Pantolon, Kazak, Elbise, Mont, Sandalet, Gömlek, Spor ayakkabı, Çanta, Bot). Hızlı çalışması için 20.000 eğitim görüntüsü kullanılır.
- **İndirme:** Veri ilk çalıştırmada yaklaşık 30 MB olarak indirilir ve `veri/` klasörüne kaydedilir. Bu klasör `.gitignore` ile depoya eklenmez.
- **Hedef modeller:** küçük bir evrişimli sinir ağı (PyTorch) ve kara kutu saldırısı için bir rastgele orman (scikit-learn).
- **Donanım:** GPU gerekmez; notebook sıradan bir dizüstü bilgisayarda yaklaşık 3 dakikada çalışır.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `talhakaraca-20259267001-ek-ornek.ipynb` | Uygulama notebook'u (çıktılarıyla birlikte) |
| `talhakaraca-20259267001-ek-ornek.html` | Notebook'un HTML hâli; tarayıcıda doğrudan açılabilir |
| `gorseller/` | Notebook'un ürettiği grafikler (PNG) |
| `requirements.txt` | Gerekli Python paketleri |

## İçerik

| # | Bölüm | Kitaptaki karşılığı | Gösterilen sonuç |
|---|-------|---------------------|------------------|
| 0 | Veri ve model | – | CNN temiz veride %89 doğruluk |
| 1 | FGSM | Rahatsız Edilmiş Panda | ε = 0,05 ile doğruluk %89 → %43; aynı büyüklükte rastgele gürültüyle %89 |
| 2 | Kategoriler arası geçişler | – | Saldırı görüntüleri benzer kategorilere iter (tişört → gömlek, sandalet → spor ayakkabı); pantolon, çanta ve bot daha dayanıklı |
| 3 | Hedefli L-BFGS saldırısı | Köpeğimde Bir Terslik Var | 9 farklı giysinin hepsi "Çanta" olarak sınıflandırılıyor |
| 4 | 1-piksel saldırısı | Bir Denizanası… Hayır, Bekle. Bir Küvet | 784 pikselden biri değiştirilerek görüntülerin %18'inde model yanılıyor |
| 5 | Kara kutu saldırısı | Gözü Bağlı Düşman | Rastgele ormanın doğruluğu %86 → %32 (ε = 0,1) |
| 6 | Adversarial eğitim ve PGD | Siber Güvenlik Perspektifi | Eğitilmiş model ε = 0,05'te FGSM'e karşı %76 doğruluk korur; ε büyüdükçe PGD korumayı aşar |

Adversarial yama ve EOT (3D kaplumbağa) fiziksel dünyaya yönelik saldırılardır ve büyük görüntü modelleri gerektirir; bu notebook'ta uygulanmadı.

## Çalıştırma

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook talhakaraca-20259267001-ek-ornek.ipynb
```

Notebook Google Colab'a da doğrudan yüklenebilir.

## Kaynaklar

- Molnar, C. *Interpretable Machine Learning*, 3. baskı (2025), Bölüm 30.
- Xiao, H., Rasul, K., Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
- Szegedy vd. (2014). Intriguing properties of neural networks.
- Goodfellow vd. (2015). Explaining and harnessing adversarial examples.
- Su vd. (2019). One pixel attack for fooling deep neural networks.
- Papernot vd. (2017). Practical black-box attacks against machine learning.
- Madry vd. (2018). Towards deep learning models resistant to adversarial attacks (PGD).
