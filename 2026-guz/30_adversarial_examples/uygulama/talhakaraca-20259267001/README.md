# Bölüm 30 Uygulaması: Kötü Amaçlı Örnekler (Adversarial Examples)

Christoph Molnar'ın *Interpretable Machine Learning* kitabının 30. bölümünde anlatılan saldırı yöntemlerinin, küçük ve basit bir veri seti üzerinde canlı kod olarak uygulanması.

- **Veri seti:** scikit-learn ile birlikte gelen 8×8 piksellik el yazısı rakamlar (`load_digits`, 1.797 görüntü, 10 sınıf). İndirme gerektirmez.
- **Hedef modeller:** küçük bir evrişimli sinir ağı (PyTorch) ve kara kutu saldırısı için bir rastgele orman (scikit-learn).
- **Donanım:** GPU gerekmez; tüm notebook sıradan bir dizüstü bilgisayarda yaklaşık yarım dakikada çalışır.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `talhakaraca-20259267001.ipynb` | Uygulama notebook'u (çıktılarıyla birlikte) |
| `talhakaraca-20259267001.html` | Notebook'un HTML hâli; tarayıcıda doğrudan açılabilir |
| `gorseller/` | Notebook'un ürettiği grafikler (PNG) |
| `requirements.txt` | Gerekli Python paketleri |
| `talhakaraca-20259267001-ek-ornek/` | Aynı yöntemlerin Fashion-MNIST (28×28 giysi görüntüleri, 10 kategori) üzerindeki uygulaması |

## İçerik

| # | Bölüm | Kitaptaki karşılığı | Gösterilen sonuç |
|---|-------|---------------------|------------------|
| 0 | Veri ve model | – | CNN temiz test verisinde ~%98 doğruluk |
| 1 | FGSM | Rahatsız Edilmiş Panda | ε = 0,2 ile doğruluk %98 → %5; aynı büyüklükte rastgele gürültüyle %90 |
| 2 | Hedefli L-BFGS saldırısı | Köpeğimde Bir Terslik Var | 9 farklı rakamın hepsi "3" olarak sınıflandırılıyor |
| 3 | 1-piksel saldırısı (diferansiyel evrim) | Bir Denizanası… Hayır, Bekle. Bir Küvet | Tek piksel değiştirerek görüntülerin ~%33'ünde model yanılıyor |
| 4 | Kara kutu saldırısı (vekil model) | Gözü Bağlı Düşman | Gradyanı olmayan rastgele ormanın doğruluğu %97 → %34 (ε = 0,2) |
| 5 | Adversarial eğitim ve PGD | Siber Güvenlik Perspektifi | Savunma FGSM'ye karşı yardımcı oluyor, daha güçlü PGD saldırısı yine işliyor |

Adversarial yama ve EOT (3D kaplumbağa) fiziksel dünyaya yönelik saldırılar olduğu ve büyük görüntü modelleri gerektirdiği için bu küçük veri setinde uygulanmamıştır.

## Çalıştırma

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook talhakaraca-20259267001.ipynb
```

Notebook Google Colab'a da doğrudan yüklenebilir; gerekli paketlerin tümü Colab'da hazır bulunur.

## Kaynaklar

- Molnar, C. *Interpretable Machine Learning*, 3. baskı (2025), Bölüm 30.
- Szegedy vd. (2014). Intriguing properties of neural networks.
- Goodfellow vd. (2015). Explaining and harnessing adversarial examples.
- Su vd. (2019). One pixel attack for fooling deep neural networks.
- Papernot vd. (2017). Practical black-box attacks against machine learning.
- Madry vd. (2018). Towards deep learning models resistant to adversarial attacks (PGD).
