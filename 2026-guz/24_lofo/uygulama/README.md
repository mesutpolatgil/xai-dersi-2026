# Bölüm 24: Bir Özelliği Dışarıda Bırakma Önemi (Leave One Feature Out - LOFO) Uygulaması

**Hazırlayan:** Yunus Emre Yazarlı  
**Öğrenci No:** 20269258002  
**Ders:** Açıklanabilir Yapay Zeka (XAI) — 2026 Güz  
**Kaynak Bölüm:** [Christoph Molnar — Interpretable Machine Learning, Chapter 24: Leave One Feature Out (LOFO) Importance](https://christophm.github.io/interpretable-ml-book/lofo.html)

## Klasör İçeriği ve Dosya Açıklamaları

Bu klasörde, 24. Bölümde ele alınan **Bir Özelliği Dışarıda Bırakma (LOFO / LOCO) Önemi** yönteminin hem interaktif web simülasyonu hem de orijinal veri setleri (`bike.csv` ve `penguins.csv`) üzerinde çalıştırılmış Python kodları ve gerçek grafik çıktıları yer almaktadır:

| Dosya Adı | Açıklama |
| :--- | :--- |
| **`24_lofo-web-uygulaması.html`** | Tarayıcıda doğrudan çalışan **İnteraktif Web Simülasyonu**. LOFO önem grafiğini, `%100` korelasyonlu kopya özellik (`temp_copy`) deneyini, interaktif özellik çıkarma ve `p`-adımlı karar ağacı yeniden eğitim animasyonunu, geriye doğru sıralı özellik seçimini (Adım 1–2–3) ve LOFO vs. PFI karşılaştırmasını içerir. |
| **`24_lofo-python-uygulaması.html`** | Kitaptaki tüm örneklerin **Python (`scikit-learn`, `pandas`, `matplotlib`, `lofo-importance`) Kodları ve Gerçek Çıktıları**. Orijinal `bike.csv` ve `penguins.csv` verileri üzerinde çalıştırılmış kod hücrelerini (`In [1]`–`In [4]`), konsol tablo çıktılarını ve gömülü grafikleri (Şekil 24.1–24.3) içerir. |
| **`bike.csv`** | Bisiklet kiralama veri seti (`728` tam gözlem, hedef: `cnt`, özellikler: `season`, `holiday`, `workday`, `weather`, `temp`, `hum`, `windspeed`, `cnt_2d_bfr`). |
| **`penguins.csv`** | Palmer penguenleri veri seti (`333` tam gözlem, hedef: `sex`, özellikler: `species`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`). |

## Uygulamaların Kapsadığı Temel Konular

1. **Şekil 24.1 — Bisiklet Kiralama Verisinde LOFO Özellik Önemi:** Her bir $j$ özelliğinin veri setinden çıkarılıp Random Forest modelinin kalan $p-1$ özellikle yeniden eğitilmesi ($\text{LOFO}_j = e_{-j} - e_{\text{orig}}$). En önemli özelliklerin `cnt_2d_bfr` ve `temp` olması, `holiday` özelliğinin ise negatif LOFO değeri alarak çıkarıldığında modeli iyileştirmesi.
2. **Şekil 24.2 — Tam Korelasyonlu Özellik (`temp_copy`) Deneyi:** Sıcaklık sütununun birebir kopyası (`temp_copy = temp`) veri setine eklendiğinde, biri çıkarıldığında yeniden eğitilen model diğerini kullandığı için her iki özelliğin de LOFO öneminin `0.00 MAE` seviyesine düşmesi (LOFO'nun kalan özelliklere koşullu olması).
3. **Şekil 24.3 — Penguen Cinsiyeti Sınıflandırmasında LOFO Analizi:** Doğruluk (Accuracy) metriği kullanılarak ($-1$ ile çarpılmış fark) hesaplanan LOFO önemlerinde `bill_depth_mm` ve `body_mass_g` özelliklerinin öne çıkması, `flipper_length_mm` ve `species` özelliklerinin ise negatif/sıfır önem alması.
4. **Geriye Doğru Sıralı Özellik Seçimi (Backward Sequential Selection) & `lofo-importance` Paketi:** Korelasyonlu özellikler varken özellikleri tek seferde silmek yerine her adımda LOFO'yu yeniden hesaplayarak tek tek çıkarma simülasyonu ve Python `lofo-importance` kütüphanesi kullanımı.

## Nasıl Çalıştırılır?

* `.html` uzantılı dosyaları bilgisayarınıza indirip herhangi bir modern web tarayıcısında (Chrome, Edge, Firefox, Safari) çift tıklayarak doğrudan açabilirsiniz.
* Python kodlarını kendi ortamınızda çalıştırmak için `24_lofo-python-uygulaması.html` içindeki **`📋 Kodu Kopyala`** butonlarını kullanabilirsiniz (`pip install numpy pandas scikit-learn matplotlib lofo-importance`).
