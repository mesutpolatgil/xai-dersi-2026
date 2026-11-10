# Bölüm 23: Permütasyon Özellik Önemi (Permutation Feature Importance - PFI) Uygulaması

**Hazırlayan:** Yunus Emre Yazarlı  
**Öğrenci No:** 20269258002  
**Ders:** Açıklanabilir Yapay Zeka (XAI) — 2026 Güz  
**Kaynak Bölüm:** [Christoph Molnar — Interpretable Machine Learning, Chapter 23: Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)

## Klasör İçeriği ve Dosya Açıklamaları

Bu klasörde, 23. Bölümde ele alınan **Permütasyon Özellik Önemi (PFI)** yönteminin hem interaktif web simülasyonu hem de orijinal veri setleri (`bike.csv` ve `penguins.csv`) üzerinde çalıştırılmış Python kodları ve gerçek grafik çıktıları yer almaktadır:

| Dosya Adı | Açıklama |
| :--- | :--- |
| **`23_feature-importance-web-uygulaması.html`** | Tarayıcıda doğrudan çalışan **İnteraktif Web Simülasyonu**. Temel PFI ve canlı sütun karıştırma (shuffle) deneyini, 4 koşullu örnekleme yönteminin (`P(X_j \| X_{-j})`) canlı 2B nokta animasyonunu ve penguen türlerine göre Grup Bazlı PFI (Şekil 23.3 & 23.4) simülasyonunu içerir. |
| **`23_feature-importance-python-uygulaması.html`** | Kitaptaki tüm örneklerin **Python (`scikit-learn`, `pandas`, `matplotlib`) Kodları ve Gerçek Çıktıları**. Orijinal `bike.csv` ve `penguins.csv` verileri üzerinde çalıştırılmış kod hücrelerini (`In [1]`–`In [5]`), konsol tablo çıktılarını, gömülü grafikleri (Şekil 23.1–23.4) ve `sklearn.inspection.permutation_importance` / `eli5` / `alibi` kütüphane örneklerini içerir. |
| **`bike.csv`** | Bisiklet kiralama veri seti (`728` tam gözlem, hedef: `cnt`, özellikler: `season`, `holiday`, `workday`, `weather`, `temp`, `hum`, `windspeed`, `cnt_2d_bfr`). |
| **`penguins.csv`** | Palmer penguenleri veri seti (`333` tam gözlem, hedef: `sex`, özellikler: `species`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`). |

## Uygulamaların Kapsadığı Temel Konular

1. **Şekil 23.1 — Bisiklet Kiralama Tahmininde SVM PFI Analizi:** Destek Vektör Makinesi (SVM) modelinde her bir özelliğin 30 tekrarlı permütasyonu sonucunda ortalama hata artış oranı ($FI_j = e_{\text{perm}} / e_{\text{orig}}$) ve %5–%95 kantil hata aralıklarının hesaplanması (`cnt_2d_bfr` ve `temp` en önemli özellikler, `holiday` en önemsiz özellik).
2. **Şekil 23.2 — Penguen Cinsiyeti Sınıflandırmasında Kara Kutu PFI Analizi:** Her penguen türü için ayrı eğitilmiş 3 Lojistik Regresyon modelini tek bir kara kutu fonksiyon olarak ele alarak Log Loss bazlı PFI hesabı (`body_mass_g` ve `species` özelliklerinin etkisi).
3. **Bölüm 23.3 — Koşullu Özellik Önemi (4 Koşullu Örnekleme Yaklaşımı):** Korelasyonlu özelliklerde gerçekçi olmayan veri noktalarını (ekstrapolasyonu) önlemek için kullanılan 4 koşullu örnekleme yönteminin (**Koşullu Alt Gruplar - Molnar 2023**, **Eşleştirme - Fisher 2019**, **Knockoff İkizler - Watson & Wright 2021** ve **Rastgele Orman Izgarası - Debeer & Strobl 2020**) canlı animasyonla karşılaştırılması.
4. **Şekil 23.3 & 23.4 — Penguen Türlerine Göre Grup Bazlı PFI (Group-wise PFI):** Tüm veri genelinde `r = 0.87` olan kütle–yüzgeç korelasyonunun tür alt kümelerinde (`Adelie r = 0.46`, `Chinstrap r = 0.64`, `Gentoo r = 0.71`) nasıl düştüğünün ve her türün kendi içinde karıştırılmasıyla elde edilen grup bazlı PFI sonuçlarının incelenmesi.

## Nasıl Çalıştırılır?

* `.html` uzantılı dosyaları bilgisayarınıza indirip herhangi bir modern web tarayıcısında (Chrome, Edge, Firefox, Safari) çift tıklayarak doğrudan açabilirsiniz; ek bir sunucu veya kurulum gerektirmez.
* Python kodlarını kendi ortamınızda çalıştırmak için `23_feature-importance-python-uygulaması.html` içindeki **`📋 Kodu Kopyala`** butonlarını kullanabilirsiniz (`pip install numpy pandas scikit-learn matplotlib scipy`).
