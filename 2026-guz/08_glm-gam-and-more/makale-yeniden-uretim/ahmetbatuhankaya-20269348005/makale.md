# Makale Yeniden Üretimi: Caruana ve ark. (2015)

**Öğrenci:** Ahmet Batuhan Kaya (20269348005)
**İlgili bölümler:** Molnar, *Interpretable Machine Learning*, Bölüm 7 (Lojistik Regresyon) ve Bölüm 8 (GLM, GAM ve ötesi)
**İzlencedeki tür:** Seçenek 2: Kıyaslama (Benchmarking)

## Makale

Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15)*, 1721–1730. https://doi.org/10.1145/2783258.2788613

Makale, çevirdiğim Bölüm 8'in etkileşimler kısmında atıf olarak geçiyor ("Bir diğer seçenek GA2M'dir").

### Makalenin önerdiği model: GA²M

GAM'e (Bölüm 8) en güçlü k ikili etkileşim eklenir:

$$g(\mathbb{E}[y]) = \beta_0 + \sum_j f_j(x_j) + \sum_{i \neq j} f_{ij}(x_i, x_j)$$

Her $f_j$ ve $f_{ij}$ ayrı ayrı çizilebildiği için model yorumlanabilir kalır.

### Makalenin iddiaları

1. **Doğruluk:** GA²M, kara kutular kadar doğru olabilir. Makalenin Tablo 2'si (test AUC):

| Model | Zatürre | 30 günlük yeniden yatış |
|---|---|---|
| Lojistik regresyon | 0,8432 | 0,7523 |
| GAM | 0,8542 | 0,7795 |
| GA²M | 0,8576 | 0,7833 |
| Rastgele orman | 0,8460 | 0,7671 |
| LogitBoost | 0,8493 | 0,7835 |

2. **Tehlikeli örüntüler:** Zatürre verisinde modeller "astım ölüm riskini azaltır" sonucunu öğreniyor. Gerçek neden: astımlı hastalar doğrudan yoğun bakıma alındığı için daha iyi tedavi görmüş. Bu modelle hasta önceliklendirilseydi astımlı hastalar zarar görebilirdi. GA²M'nin şekil grafiğinde bu örüntü açıkça görülüyor ve düzeltilebiliyor (terimi çıkarmak ya da uzman bilgisiyle yeniden çizmek). Makale kronik akciğer hastalığı ve göğüs ağrısı için de benzer şüpheler dile getiriyor.

## Yeniden üretim tasarımı

Makalenin verileri hastane verisi olduğu için kamuya açık değil. Makalenin **ikinci görevini (30 günlük yeniden yatış)** kamuya açık bir veride yeniden ürettim.

| | Makale | Bu çalışma |
|---|---|---|
| Veri | Büyük bir hastanenin kayıtları (2011–2013) | ABD'deki 130 hastanenin diyabet hastası yatışları, 1999–2008 (Strack ve ark., 2014) |
| Hasta sayısı | ~296.000 | 71.518 (her hastanın ilk yatışı) |
| Öznitelik | ~4.000 | 20 |
| Yeniden yatış oranı | %8,91 | %8,80 |
| GA²M uygulaması | Yazarların kendi uygulaması | InterpretML EBM (aynı ekibin güncel uygulaması) |
| Eğitim/test | Zamana göre ayrım | %70/%30 rastgele, tabakalı |

Ön işlem notları:
- Bağımsızlık varsayımı için her hastanın yalnızca ilk yatışı tutuldu (Strack ve ark. gibi).
- Tanı kodları (ICD-9) Strack ve ark.'ın gruplamasıyla 9 gruba indirildi.
- Taburcu ve kabul kodları, UCI'deki kod tablosuna göre okunur kategorilere çevrildi.
- AUC için test kümesinde 200 bootstrap ile %95 güven aralığı hesaplandı.

## Bulgular

### 1. Doğruluk

| Model | Test AUC | %95 GA |
|---|---|---|
| Lojistik regresyon | 0,644 | 0,634 – 0,658 |
| GAM (EBM, etkileşimsiz) | 0,643 | 0,629 – 0,657 |
| GA²M (EBM, 10 etkileşim) | 0,647 | 0,635 – 0,661 |
| Rastgele orman | 0,651 | 0,637 – 0,664 |
| Gradient boosting | 0,650 | 0,636 – 0,663 |

- **Destekleniyor:** Yorumlanabilir modeller kara kutularla aynı düzeyde; güven aralıkları tamamen örtüşüyor.
- **Farklılaşıyor:** Makalede GAM ve GA²M lojistik regresyonu yaklaşık 3 puan geçiyordu; bizde fark yok. En olası neden, verimizin çok daha küçük ve öznitelik açısından fakir olması (20'ye karşı ~4.000). Doğrusal olmayan etkilerin yakalayacağı fazla sinyal kalmıyor. Mutlak AUC'lerin düşük olması da aynı nedenden; bu veri için literatürde de benzer değerler raporlanıyor.

### 2. Şekil fonksiyonları: doğrusallık nerede çöküyor?

- **Hastanede kalış süresi:** Risk 9–10 güne kadar artıyor, sonra belirgin biçimde düşüyor (ters-U). Lojistik regresyon tek bir eğimle sürekli artış varsayıyor.
- **Laboratuvar işlemi sayısı:** Yaklaşık 75 işleme kadar hafif artış, sonrasında keskin düşüş.
- **Önceki yatış sayısı:** En güçlü sayısal yordayıcı; etkisi basamaklı ve doymaya yakın.
- **Yaş:** 20 yaş altında belirgin biçimde düşük risk, sonra yavaş artış.

### 3. Tehlikeli örüntü: "astım riski azaltır"ın bu verideki karşılığı

GA²M'ye göre en önemli öznitelik **taburcu durumu**. Model, taburculukta **vefat eden** (1.084 hasta, %0 yeniden yatış) ve **hospise giden** (461 hasta, %3,5) hastaları en düşük riskli grup olarak öğreniyor.

Bu, veride doğru ama kavramsal olarak yanıltıcı: Yeniden yatmamak iyileşmek demek değil. Ölüm, yeniden yatışı imkânsız kılan rakip bir olay (competing risk). Ayrıca "vefat" bilgisi sonucun kendisini taşıyan bir sızıntı (leakage). Riski "hastanın ne kadar ağır olduğu" diye okuyan bir karar verici, en ağır hastaları en düşük riskli sanardı. Makaledeki astım bulgusuyla aynı ders: **model veriyi açıklar, dünyayı değil.**

Bölüm 7 ile bağlantı: "Vefat" kategorisinde hiç yeniden yatış olmadığı için bu bir **tam ayrışma** durumu. Lojistik regresyonda katsayı −6,94 (odds oranı 0,001) çıkıyor ve yalnızca L2 cezası sayesinde sonlu kalıyor. EBM'nin katkısı −4,7'de kalıyor, çünkü boosting küçük adımlarla öğreniyor.

**Düzeltme:** Makalenin önerdiği gibi bu hastaları modelden çıkardım (Strack ve ark. da aynısını yapıyor). Vefat/hospis hariç test kümesinde AUC 0,640'tan 0,641'e çıkıyor; doğruluk korunurken kavramsal olarak hatalı kural ortadan kalkıyor.

### 4. Etkileşimler

GA²M'nin seçtiği en güçlü etkileşim: laboratuvar işlemi sayısı × ilaç sayısı. Seçilen 10 etkileşimin 4'ü taburcu durumunu içeriyor; bu da taburcu durumunun modeldeki baskın rolünü doğruluyor.

## Sonuç

| İddia | Bu çalışmada |
|---|---|
| Yorumlanabilir model kara kutu kadar doğru olabilir | **Destekleniyor** |
| GAM/GA²M lojistik regresyonu geçer | **Desteklenmiyor** (bu veri ve öznitelik setinde) |
| Yorumlanabilirlik tehlikeli örüntüleri görünür kılar | **Destekleniyor** (taburcu durumu = vefat/hospis) |

## Sınırlılıklar

- Farklı hasta kitlesi (diyabet), farklı dönem ve çok daha az öznitelik.
- Tek bir eğitim/test ayrımı; makale zamana göre ayrım kullanıyor.
- EBM, makaledeki GA²M ile aynı fikri uyguluyor ama birebir aynı algoritma değil (etkileşim seçimi ve öğrenme yöntemi farklı).

## Çalıştırma

```bash
pip install numpy pandas matplotlib scikit-learn interpret
jupyter notebook analiz.ipynb
```

Veri ilk çalıştırmada otomatik indirilir (yaklaşık 19 MB) ve aynı klasöre `diabetic_data.csv` olarak kaydedilir. Bu nedenle `veri.csv` ayrıca repoya eklenmedi. Asıl kaynak: UCI Machine Learning Repository, "Diabetes 130-US Hospitals for Years 1999-2008".

## Kaynaklar

- Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission. *KDD '15*, 1721–1730.
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014, 781670.
- Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. arXiv:1909.09223.
- Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate Intelligible Models with Pairwise Interactions. *KDD '13*.
- Molnar, C. *Interpretable Machine Learning*. https://christophm.github.io/interpretable-ml-book/
