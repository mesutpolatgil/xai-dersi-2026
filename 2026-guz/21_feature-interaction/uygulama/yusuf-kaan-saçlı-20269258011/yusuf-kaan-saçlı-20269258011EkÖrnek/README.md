# Özellik Etkileşimi (Bölüm 21) ve Fonksiyonel Ayrıştırma (Bölüm 22) — Jupyter Notebook Uygulaması

Bu proje, **Christoph Molnar**'ın *Interpretable Machine Learning* kitabının **21. Bölümü (Özellik Etkileşimi / Feature Interaction)** ve **22. Bölümü (Fonksiyonel Ayrıştırma / Functional Decomposition)** konularını hiçbir hazır XAI paketi (`iml`, `shap`, `alibi` vb.) kullanmadan, tamamen **`numpy` ile sıfırdan** kodlayarak adım adım açıklayan Türkçe bir Jupyter Notebook uygulamasıdır.

- **Hazırlayan:** Yusuf Kaan Saçlı (`20269258011`)
- **Kaynak Kitap:** Christoph Molnar, *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2022), Bölüm 21 ve Bölüm 22 — <https://christophm.github.io/interpretable-ml-book/>
- **Lisans:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Klasör İçeriği

| Dosya | Açıklama |
|---|---|
| `YusufKaanSaçlı-20269258011.ipynb` | Tüm hücreleri çalıştırılmış, çıktıları ve grafikleri içinde barındıran ana Jupyter Notebook dosyası |
| `YusufKaanSaçlı-20269258011.html` | `jupyter nbconvert --to html` ile üretilmiş, tüm grafikleri gömülü (tek dosya, çevrimdışı açılabilir) HTML raporu |
| `main.py` | `jupyter nbconvert --to script` ile üretilmiş, tüm Türkçe açıklamaları yorum satırı (`#`) olarak içeren eşdeğer Python betiği |
| `penguins.csv` | Bölüm 0 ve Bölüm 21.3'te kullanılan Palmer Penguenleri veri seti |
| `requirements.txt` | Test edilen ortamın kurulu Python paket sürümleri |
| `README.md` | Kurulum, çalıştırma ve dışa aktarma kılavuzu |

---

## Windows Komut İstemi (CMD) ile Kurulum ve Çalıştırma Adımları

### 1. Klasöre Geçiş ve Sanal Ortam Kurulumu

Windows'ta **Komut İstemi (`cmd.exe`)** açın ve proje klasörüne gidin:

```cmd
cd calisma\uygulama-notebook\YusufKaanSaçlı-20269258011
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Not:** `requirements.txt` dosyası Python `3.13.14` ortamında test edilen tam sürümleri içerir. Farklı bir Python 3.10+ alt sürümü kullanıyorsanız ve sürüm uyuşmazlığı alırsanız paket adlarını sürüm sabitlemesi olmadan kurabilirsiniz:
> `pip install numpy pandas scikit-learn matplotlib seaborn jupyter nbconvert ipykernel`

### 2. HTML Çıktısını Doğrudan Görüntüleme (Kurulumsuz)

Hazır üretilmiş olan `YusufKaanSaçlı-20269258011.html` dosyasına çift tıklayarak herhangi bir tarayıcıda doğrudan açabilirsiniz; tüm grafikler dosyanın içine gömülüdür.

### 3. Notebook'u Tarayıcıda Etkileşimli Açma

```cmd
jupyter notebook YusufKaanSaçlı-20269258011.ipynb
```
*(Alternatif olarak JupyterLab kullanıyorsanız: `jupyter lab YusufKaanSaçlı-20269258011.ipynb`)*

### 4. Notebook'u Komut Satırından Baştan Sona Çalıştırma (`ipynb` Güncelleme)

Tüm hücreleri sırasıyla çalıştırıp çıktıları `.ipynb` dosyasının içine kaydetmek için:

```cmd
jupyter nbconvert --to notebook --execute --inplace YusufKaanSaçlı-20269258011.ipynb
```

### 5. Tek Dosyalık Gömülü HTML Raporu Üretme

Çalıştırılmış notebook'tan tüm görselleri içine gömülü `YusufKaanSaçlı-20269258011.html` dosyasını üretmek için:

```cmd
jupyter nbconvert --to html YusufKaanSaçlı-20269258011.ipynb
```

### 6. Eşdeğer Python Betiği (`main.py`) Üretme ve Çalıştırma

Notebook'taki markdown hücrelerini yorum satırına dönüştürerek `main.py` dosyasını üretmek ve uçtan uca hatasız çalıştığını doğrulamak için:

```cmd
jupyter nbconvert --to script YusufKaanSaçlı-20269258011.ipynb --output main
python main.py
```

---

## Kapsanan Konular Özeti

1. **Bölüm 0 — Veri Hazırlığı:** Palmer Penguenleri (`penguins.csv`) keşifçi analizi, eksik değer temizliği ve `P(dişi)` hedef değişkeninin tanımlanması.
2. **Bölüm 21 — Özellik Etkileşimi:**
   - **1.1** Ev fiyatı tahmini (Tablo 1 etkileşimsiz ve Tablo 2 etkileşimli ayrıştırma: `150.000 + 50.000 + 100.000 + 100.000 = 400.000`).
   - **1.2** Sıfırdan `numpy` ile 1B/2B Kısmi Bağımlılık (PD), ikili $H^2_{jk}$ ve toplam $H^2_j$; toplamsal ($H \approx 0$) ve saf çarpım ($H \approx 1$) doğrulama testleri.
   - **1.3** Penguen cinsiyet tahmini (Rastgele Orman, Gradyan Artırma, Lojistik Regresyon), toplam ve ikili H grafikleri, türlere göre etkileşimler, 2B PDP ısı haritası.
   - **1.4** Örneklem boyutu ($n \in \{20, 50, 100, 150\}$) kararsızlık analizi (20 tekrar kutu grafiği) ve hesap maliyeti tablosu ($2n^2$, $3n^2$).
   - **1.5** Normalize edilmemiş $H^*_{jk}$ istatistiği ve küçük paydada sahte etkileşim demosu.
3. **Bölüm 22 — Fonksiyonel Ayrıştırma:**
   - **2.1** $\hat{f}(x_1, x_2) = 2 + e^{x_1} - x_2 + x_1 \cdot x_2$ fonksiyonunun $\hat{f}_0 \approx 3{,}18$, $\hat{f}_1$, $\hat{f}_2$, $\hat{f}_{1,2}$ bileşenlerine ayrıştırılması ve doğrulanması.
   - **2.2** Kısıtsız ayrıştırmada benzersizlik problemi (etkinin ana etkiden etkileşime kaydırılması).
   - **2.3** Hooker (2004) Fonksiyonel ANOVA (fANOVA) bileşenleri ve üç aksiyomun (sıfır ortalama, diklik, varyans ayrıştırması) sayısal sınanması.
   - **2.4** Sıfırdan `numpy` ile 1B/2B Birikimli Yerel Etkiler (ALE) ve korelasyonlu özelliklerde ($\rho \in \{0, 0{,}6, 0{,}9\}$) PDP ekstrapolasyon yanılgısı ile karşılaştırma.
   - **2.5** Aşağıdan yukarıya yaklaşım: Spline + Ridge ile Genelleştirilmiş Toplamsal Model (GAM) ve ikili etkileşimli GA2M karşılaştırması.
