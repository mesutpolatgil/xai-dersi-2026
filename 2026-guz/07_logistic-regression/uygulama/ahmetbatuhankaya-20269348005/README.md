# Bölüm 7 – Lojistik Regresyon: Uygulama

**Öğrenci:** Ahmet Batuhan Kaya (20269348005)
**Kaynak bölüm:** Molnar, *Interpretable Machine Learning*, Bölüm 7 – Logistic Regression

## Amaç

Lojistik regresyonun yorum araçlarını (log-odds, odds oranı, güven aralığı, tekil tahmin açıklaması) **zorlu bir gerçek dünya veri setinde** uygulamak ve yöntemin bilinen zayıflıklarını canlı olarak göstermek.

## Veri setleri

| Veri | Kaynak | Kullanım |
|---|---|---|
| Palmer Penguins (Chinstrap) | Horst, Hill & Gorman (2020), `palmerpenguins` | Kitaptaki tablonun yeniden üretimi |
| Simüle tümör verisi | Kitaptaki kod | Doğrusal ve lojistik regresyonun karşılaştırılması |
| German Credit (1.000 başvuru) | Hofmann (1994), UCI Machine Learning Repository | Asıl uygulama |

German Credit neden zorlu: dengesiz sınıflar (%30 temerrüt), çok seviyeli kategorik öznitelikler, az gözlemli kategoriler ve bilinen veri kalitesi sorunları.

## İçerik

| Adım | Ne yapılıyor | Kitapla bağlantısı |
|---|---|---|
| 1 | Penguen modeli `statsmodels` ile yeniden kuruluyor; odds oranları kitapla birebir aynı | Bölüm 7 örneği |
| 2 | Tümör verisinde doğrusal ve lojistik regresyon karşılaştırılıyor | "Sınıflandırma için doğrusal regresyon kullanmayın" |
| 3–4 | Kredi modeli, odds oranı grafiği, aynı odds oranının farklı başlangıç risklerinde etkisi, tekil başvuru açıklaması | Yorumlama |
| 5 | Odds oranı ile risk oranının farkı | Yorum tuzağı |
| 6 | Tam ayrışma: katsayı ve standart hatanın patlaması, ceza ile çözüm | Sınırlılıklar: complete separation |
| 7 | Non-collapsibility simülasyonu | Modeller arası karşılaştırma sorunu |
| 8 | Test verisinde AUC ve kalibrasyon | "Olasılıklar kalibre mi?" |

## Temel bulgular

- **Kitapla tutarlılık:** Penguen modelinde gaga uzunluğu, normal tombulluk ve en tombul grup için odds oranları kitaptaki gibi 0,59; 1,92 ve 0,43. Gaga uzunluğu için odds oranı 1'den küçük olduğu hâlde kitabın metni "artırır" diyor; bu bir yorum hatası.
- **Çarpımsal etki:** Vadenin 1 ay uzaması temerrüt odds'unu 1,031 katına, 12 ay uzaması 1,45 katına çıkarıyor. Aynı odds oranı (2) başlangıç riski %5 iken olasılığı 4,5 puan, %50 iken 16,7 puan artırıyor.
- **Odds oranı ≠ risk oranı:** Eksi bakiyeli ve bakiyesi bilinmeyen müşteriler karşılaştırıldığında risk oranı 4,2, odds oranı 7,3.
- **Tam ayrışma:** Sınıfları kusursuz ayıran bir öznitelik eklendiğinde katsayı 26,6, standart hatası 42.000'in üzerinde çıkıyor; ceza (L2) azaldıkça katsayı sınırsız büyüyor.
- **Non-collapsibility:** Bağımsız bir z değişkeni eklendiğinde doğrusal regresyonda x'in katsayısı değişmezken (1,01 → 1,01) lojistik regresyonda 0,65'ten 1,08'e çıkıyor.
- **Sezgiye aykırı katsayılar:** "Kritik kredi geçmişi" riski azaltıyor, "tamamen ödemiş" riski artırıyor gibi görünüyor. Bu, seçilim yanlılığına veya veri setindeki bilinen kodlama hatalarına (Grömping, 2019) işaret ediyor. Yorumlanabilir model bu sorunu görünür kılıyor.
- **Performans:** Test AUC yaklaşık 0,79.

## Çalıştırma

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn
python main.py
```

Ya da VS Code'da her `# %%` hücresini **Run Cell** ile çalıştırın. Veriler internetten otomatik indirilir; grafikler `sekiller/` klasörüne kaydedilir; rastgelelik `seed=42` ile sabitlenmiştir. 6. adımdaki `PerfectSeparationWarning` ve yakınsama uyarıları beklenen davranıştır ve deneyin parçasıdır.

## Kaynaklar

- Molnar, C. *Interpretable Machine Learning*. https://christophm.github.io/interpretable-ml-book/
- Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.
- Grömping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set. *Reports in Mathematics, Physics and Chemistry*, Beuth University of Applied Sciences Berlin.
- Mood, C. (2010). Logistic Regression: Why We Cannot Do What We Think We Can Do, and What We Can Do About It. *European Sociological Review*, 26(1), 67–82.
- Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data.
