# Makale Yeniden Üretimi: Krishna ve ark. (2022) — Uyuşmazlık Problemi

**Öğrenci:** Berkay Çongar (20269258007)
**İlgili bölüm:** Molnar, *Interpretable Machine Learning*, Bölüm 32 (Yorumlanabilirlik Yöntemlerinin Değerlendirilmesi)
**İzlencedeki tür:** Seçenek 2: Kıyaslama (Benchmarking)

## Makale

Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S. ve Lakkaraju, H. (2022). *The Disagreement Problem in
Explainable Machine Learning: A Practitioner's Perspective.* arXiv:2202.01602. https://arxiv.org/abs/2202.01602

### Bölüm 32 ile bağlantısı

Bölüm 32'nin çıkış noktası şu: Gerçek veride bir açıklamanın temel gerçekliği yoktur. Bu makale aynı sorunun pratikteki sonucunu
ölçüyor. Aynı model ve aynı örnek için farklı açıklama yöntemleri farklı açıklamalar veriyor. Temel gerçeklik olmadığı için
hangisinin haklı olduğu da söylenemiyor. Yazarların görüştüğü 25 veri bilimcinin 21'i (%84) bu durumla karşılaşmış. Çevrimiçi çalışmadaki katılımcıların yarısı da
çelişkiyi kişisel sezgisiyle çözdüğünü söylemiş. Makalenin ölçüm aracı, Bölüm 32'deki "tutarlılık" fikrinin yöntemler arasına
taşınmış hâli: Aynı açıklanacak şey için farklı yöntemler ne kadar benzer açıklama üretiyor?

### Makalenin yöntemi

* **Açıklama yöntemleri (6):** LIME, KernelSHAP, Vanilla Gradient (Grad), Gradient×Input, Integrated Gradients (IG), SmoothGrad.
  Gradyan yöntemleri yalnızca türevlenebilir modellere (lojistik regresyon, sinir ağı) uygulanıyor.
* **Uyum metrikleri (6):** $\text{top}_k(E)$, mutlak atfı en büyük $k$ öznitelik olmak üzere:

| Metrik | Tanım |
|---|---|
| Öznitelik uyumu (FA) | $\lvert \text{top}_k(E_a) \cap \text{top}_k(E_b) \rvert / k$ |
| Sıra uyumu (RA) | Aynı sırada aynı öznitelik bulunan konum sayısı $/ k$ |
| İşaret uyumu (SA) | Ortak top-$k$ öznitelikten işareti de aynı olanların sayısı $/ k$ |
| İşaretli sıra uyumu (SRA) | Hem sırası hem işareti aynı olanların sayısı $/ k$ |
| Sıra korelasyonu (RC) | Tüm özniteliklerde $\lvert E_a \rvert$ ve $\lvert E_b \rvert$ arasında Spearman $\rho$ |
| İkili sıra uyumu (PRA) | Öznitelik çiftlerinden göreli sırası aynı olanların oranı |

* **Tablo verileri ve modeller:** COMPAS (7 öznitelik, 4.937 kişi, %70/%30) ve German Credit (20 öznitelik, 1.000 kişi, %80/%20);
  lojistik regresyon (LR), 50 ağaçlı gradyan artırma (GBT) ve rastgele orman (RF), 4 gizli katmanlı (50-100-100-50, ReLU) sinir ağı (NN).
  Makale ayrıca metin ve görüntü deneyleri içeriyor; biz yalnızca tablo verisi kısmını yeniden ürettik.
* **Ayarlar:** LIME 3.000 örnek; KernelSHAP COMPAS'ta 128 koalisyon (7 öznitelik için tam sayım), German Credit'te 3.000;
  IG 1.500 adım; SmoothGrad 1.500 örnek. Metrikler test örnekleri üzerinden ortalanıyor.

### Makalenin tablo verisi için iddiaları

* **İ1.** Yöntemler sıklıkla anlaşmıyor; metriklerin çoğu 1'in belirgin biçimde altında.
* **İ2.** Sıra uyumu (RA) ve işaretli sıra uyumu (SRA), $k$ arttıkça düşüyor.
* **İ3.** Öznitelik sayısı arttıkça uyuşmazlık artıyor (German Credit, COMPAS'tan daha uyumsuz).
* **İ4.** Daha karmaşık modelde (NN) uyuşmazlık, lojistik regresyondakine benzer ya da daha fazla.
* **İ5.** Bazı çiftler güçlü biçimde uyumlu (Grad–SmoothGrad, Grad×Input–IG), bazıları sürekli uyumsuz (örneğin Grad–IG).

## Yeniden üretim tasarımı

| | Makale | Bu çalışma |
|---|---|---|
| Veri A | COMPAS, 7 öznitelik, 4.937 kişi, 1.482 test örneği | ProPublica'nın standart filtreleri → 6.172 kişi; katmanlı %80 alt örneklem → **4.937**; %70/%30 → **1.482** test örneği. Hepsi açıklandı. |
| Etiket | COMPAS risk puanından (ayrıntı yok) | `score_text == "High"` → yüksek risk (%18,5) |
| Veri B | German Credit (20 öznitelik) | **Taiwan kredi temerrüdü** (kendi verimiz; 23 öznitelik, 30.000 müşteri, güçlü çoklu bağlantı). Rastgele 300 test örneği açıklandı. |
| Modeller | LR; GBT ve RF (50 ağaç); NN 50-100-100-50 ReLU | Aynı mimariler (scikit-learn); NN: Adam, erken durdurma |
| LIME | 3.000 örnek; çekirdek genişliği raporlanmamış | 3.000 örnek; varsayılan çekirdek; sürekli öznitelikler, örneğin çevresinde örnekleme |
| KernelSHAP | 128 (COMPAS) / 3.000 | Aynı; arka plan (raporlanmamış): eğitim verisinin 25 merkezli k-ortalamalar özeti |
| IG | 1.500 adım; taban raporlanmamış | 1.500 adım; taban = eğitim ortalaması |
| SmoothGrad | 1.500 örnek; gürültü raporlanmamış | 1.500 örnek; σ = her özniteliğin 1.–99. yüzdelik aralığının %10'u |
| Açıklanan nicelik | Raporlanmamış | Tüm yöntemlerde $P(y = 1 \mid x)$ |
| Gradyanlar | Raporlanmamış (büyük olasılıkla PyTorch) | scikit-learn ağırlıklarından elle geri yayılım; sonlu farklarla doğrulandı (en büyük fark ≈ 1·10⁻¹¹) |
| Ek | — | Her karşılaştırmaya **rastgele açıklama** eklendi; böylece metriklerin şans düzeyi görülüyor |

**4.937 sayısı üzerine bir not:** Filtrelerden sonra 6.172 kişi kalıyor; 4.937 tam olarak bunun %80'i. Makale büyük olasılıkla %80/%20
bölünmüş bir kopyanın eğitim kısmını "veri seti" olarak kullanmış. Aynı sayılara ulaşmak için bu adımı taklit ettik.

## Bulgular

Tüm sayılar `analiz.ipynb` dosyasındadır; aşağıda $k = 5$'teki ortalamalar ve parantez içinde şans düzeyleri veriliyor.

### 1. Ön kontrol: model doğrulukları

| Model | Makale | Bu çalışma |
|---|---|---|
| LR | 0,84 | 0,842 |
| GBT | 0,83 | 0,846 |
| RF | 0,82 | 0,823 |
| NN | 0,84 | 0,844 |

Dört modelin doğruluğu makaleyle iki yüzde puanı içinde örtüşüyor. Ön işleme raporlanmamış olsa da kurulumumuzun makaleninkine yakın
olduğunu gösteriyor.

### 2. Makalenin iddiaları

| | COMPAS LR | COMPAS NN | Taiwan LR | Taiwan NN |
|---|---|---|---|---|
| FA@5 | 0,888 (0,713) | 0,877 (0,713) | 0,653 (0,215) | 0,455 (0,206) |
| RA@5 | 0,639 (0,142) | 0,387 (0,145) | 0,500 (0,044) | 0,152 (0,042) |
| SA@5 | 0,591 (0,355) | 0,525 (0,354) | 0,492 (0,107) | 0,332 (0,102) |
| SRA@5 | 0,454 (0,072) | 0,287 (0,074) | 0,417 (0,020) | 0,122 (0,022) |
| RC | 0,885 (≈0) | 0,731 (≈0) | 0,821 (≈0) | 0,395 (≈0) |
| PRA | 0,910 (0,496) | 0,811 (0,497) | 0,858 (0,498) | 0,647 (0,496) |

| İddia | Sonucumuz | Karar |
|---|---|---|
| **İ1.** Yöntemler sıklıkla anlaşmıyor | Her dört durumda da RA, SA ve SRA 1'in çok altında. En kötü durumda (Taiwan NN) iki yöntemin ilk beş sırasından ortalama birinden azı aynı (RA@5 = 0,152). | Destekleniyor |
| **İ2.** RA ve SRA, $k$ arttıkça düşüyor | COMPAS NN'de $k = 1$'den 7'ye RA ≈ 0,61'den ≈ 0,40'a, SRA ≈ 0,45'ten ≈ 0,29'a iniyor. LR'de düşüş daha zayıf. | Destekleniyor |
| **İ3.** Öznitelik arttıkça uyuşmazlık artıyor | Taiwan (23 öznitelik) COMPAS'tan (7) belirgin biçimde daha uyumsuz. Şans düzeyine göre düzeltilmiş değerlerde de [(değer − şans)/(1 − şans)] fark sürüyor: NN'de FA 0,57 → 0,31; RA 0,28 → 0,11. | Destekleniyor, şerhle (aşağıda) |
| **İ4.** Karmaşık model daha uyumsuz | Her iki veride de NN, LR'den daha uyumsuz: RC COMPAS'ta 0,885 → 0,731, Taiwan'da 0,821 → 0,395. Ağaç modellerinde LIME–KernelSHAP RC en düşük RF'de (COMPAS 0,579; Taiwan 0,256). | Destekleniyor |
| **İ5.** Grad–SmoothGrad ve Grad×Input–IG uyumlu, Grad–IG uyumsuz | Grad–SmoothGrad RC: COMPAS NN 0,934, Taiwan NN 0,666, LR'de her iki veride **tam 1,000**. Grad×Input–IG: 0,863 / 0,715, LR'de **1,000**. Grad–IG: 0,620 / 0,304 (LR: 0,827 / 0,710). | Destekleniyor ve nedeni türetildi (3.1) |

**İ3'e şerh:** Mutlak FA değerlerini karşılaştırmak yanıltıcı. Rastgele iki açıklamanın beklenen FA@k'sı $k/d$'dir: COMPAS'ta 5/7 ≈ 0,71,
Taiwan'da 5/23 ≈ 0,22. Makale öznitelik sayısı farklı veri setlerini karşılaştırırken bunu hesaba katmıyor. Bizim şansa göre düzeltilmiş
karşılaştırmamız iddiayı yine destekliyor ama farkın bir kısmının yalnızca "daha çok öznitelik arasından seçim yapmaktan" geldiğini gösteriyor.

### 3. Makalenin ötesine geçen bulgular

#### 3.1 Uyuşmazlığın bir kuralı var: Yöntemler iki farklı soruyu yanıtlıyor

Makale bazı çiftlerin uyumlu, bazılarının uyumsuz olduğunu gözlemliyor ama bunu bir kuralla açıklamıyor. Lojistik regresyonda
($p = \sigma(w^\top x + b)$, $x$ eğitim ortalamasına göre) yöntemlerin ne hesapladığını yazınca kural kendiliğinden çıkıyor:

| Aile | Yöntem | Lojistik regresyonda | Sorduğu soru |
|---|---|---|---|
| **Eğim** | Grad | $p(1-p)\,w_j$ | "Bu özniteliği biraz değiştirirsem tahmin ne kadar değişir?" |
| | SmoothGrad | $\mathbb{E}_\epsilon[p(1-p)]\,w_j$ | |
| | LIME | yerel eğim $\approx c\,w_j$ | |
| **Katkı** | Grad×Input | $p(1-p)\,w_j\,x_j$ | "Bu kişinin tahminine bu öznitelik ne kadar kattı?" |
| | IG (taban 0) | $w_j x_j \cdot [\sigma(w^\top x + b) - \sigma(b)] / (w^\top x)$ | |
| | KernelSHAP | $\approx w_j (x_j - \mathbb{E}[x_j])$ (logit ölçeğinde tam eşit) | |

İki aile arasındaki fark tek bir çarpandır: $x_j$. Buradan iki öngörü çıkar ve ikisi de tutuyor:

1. **Aile içinde uyum yüksek, aileler arasında düşük.** Lojistik regresyonda aile içi çiftlerin sıra korelasyonu her iki veride de tam
   1,000. COMPAS NN'de aile içi RC 0,74–0,94, aileler arası 0,57–0,69; Taiwan NN'de aile içi 0,44–0,76, aileler arası 0,04–0,59.
   Isı haritalarında (yöntemler aile sırasıyla dizildiğinde) iki blok açıkça görülüyor.
2. **Aileler arasında işaret uyumu yaklaşık yarı yarıya, hatta şansın altında.** Eğimin işareti $w_j$'ye, katkının işareti
   $w_j x_j$'ye bağlıdır; kişi ortalamanın altındaysa ($x_j < 0$) zıt işaret çıkar. COMPAS NN'de aileler arası SA@5 0,26–0,31,
   rastgele açıklamanınki 0,35–0,36; aile içinde ise 0,82–0,94.

Sonuç: Makalenin ölçtüğü uyuşmazlığın önemli bir kısmı yöntemlerin hatası değil, iki farklı sorunun doğru cevaplarının birbirinden
farklı olması. Aynı aileden yöntemler arasındaki uyuşmazlık ise gerçek bir sorun. Örneğin Taiwan NN'de Grad ile SmoothGrad arasında
RC yalnızca 0,666; doğrusal olmayan bir modelde eğim noktadan noktaya değişiyor.

#### 3.2 "Hangisi haklı?" sorusunun cevabı: soruya göre değişiyor

Bölüm 32'nin "temel gerçekliğe benzer bir şey ancak simüle edilmiş veriyle üretilebilir" önerisini uyguladık. Veriyi bilinen bir
lojistik modelden ürettik: 8 öznitelik, $\beta = (1{,}5;\ -1{,}2;\ 1{,}0;\ -0{,}8;\ 0{,}6;\ 0;\ 0;\ 0)$. Böylece iki temel gerçekliği
de biliyoruz: katkı ($\beta_j x_j$) ve duyarlılık ($\beta_j$). Her yöntemin bunlarla uyumu (Spearman sıra korelasyonu;
bu deneyde öznitelikler arasında korelasyon yok):

| | LIME | Grad | SmoothGrad | KernelSHAP | Grad×Input | IG | Rastgele |
|---|---|---|---|---|---|---|---|
| LR: duyarlılığa uyum | **0,976** | **0,976** | **0,976** | 0,694 | 0,705 | 0,705 | −0,03 |
| LR: katkıya uyum | 0,807 | 0,807 | 0,807 | **0,896** | **0,909** | **0,909** | −0,03 |
| NN: duyarlılığa uyum | **0,919** | **0,799** | **0,880** | 0,674 | 0,488 | 0,637 | −0,03 |
| NN: katkıya uyum | 0,788 | 0,707 | 0,766 | **0,882** | **0,705** | **0,851** | −0,03 |

Eğim ailesi duyarlılık gerçeğine, katkı ailesi katkı gerçeğine daha yakın; bu, lojistik regresyonda da sinir ağında da geçerli.
Uygulayıcı için pratik sonuç: Yöntem seçmeden önce soruyu seçmek. Örneğin "başvuru neden reddedildi?" bir katkı sorusu,
"müşteri neyi değiştirirse sonuç değişir?" bir duyarlılık sorusu.

#### 3.3 Uyum, doğruluk değildir

Aynı simülasyonda, gerçek etkisi olan iki özniteliği ($x_1$, $x_2$) gerçek etkisi sıfır olan iki "sahte vekil" öznitelikle
($x_6$, $x_7$) giderek artan $\rho$ ile ilişkilendirdik. Her $\rho$ için 3 tekrar yaptık. Lojistik regresyon, katkı ailesi:

| $\rho$ | 0 | 0,5 | 0,8 | 0,95 | 0,99 |
|---|---|---|---|---|---|
| Yöntemler kendi aralarında (RC) | 0,979 | 0,980 | 0,977 | 0,974 | 0,972 |
| Yöntemler ile temel gerçeklik (RC) | 0,905 | 0,909 | 0,890 | 0,826 | 0,676 |
| Sahte vekillere giden atıf payı | 0,045 | 0,045 | 0,060 | 0,109 | 0,191 |
| Modelin sahte katsayıları $\lvert\hat\beta_6\rvert + \lvert\hat\beta_7\rvert$ | 0,236 | 0,203 | 0,299 | 0,575 | 1,025 |

Korelasyon arttıkça model gerçekte etkisi olmayan özniteliklere ağırlık veriyor. Açıklama yöntemleri modele sadık oldukları için bunu
aynen raporluyor. Yöntemler kendi aralarında neredeyse tam uyumlu kalırken hep birlikte temel gerçeklikten uzaklaşıyor. Uyuşmazlık
metrikleri bu hatayı göremez. Bu, Bölüm 32'deki **doğruluk** (açıklama görülmemiş veriyi/gerçeği ne kadar iyi tahmin ediyor) ile
**sadakat** (açıklama modele ne kadar iyi yaklaşıyor) ayrımının somut bir örneği.

## Tartışma

* Makalenin tablo verisi için ana mesajı, kendi verisinde de kendi verimizde de güçlü biçimde yeniden üretildi. Çoklu bağlantılı ve
  daha çok öznitelikli Taiwan verisinde uyuşmazlık daha da belirgin.
* Uyuşmazlık tek tip bir sorun değil. Bir kısmı **tahmin edilen niceliğin farkından** geliyor (eğim ve katkı). Bu durumda doğru
  davranış, soruya uygun aileyi seçmek. Bir kısmı ise aynı soruyu soran yöntemlerin gerçekten anlaşamamasından geliyor. Doğrusal
  olmayan modelde Grad ile SmoothGrad'in ayrışması ya da LIME'ın yüksek boyuttaki örnekleme sorunları buna örnek. Asıl ölçülmesi
  gereken bu ikinci tür.
* Uyum metrikleri bir **tutarlılık** ölçüsüdür, doğruluk ölçüsü değil. Bölüm 32'nin söylediği gibi doğruluğu ancak temel gerçekliğin
  bilindiği simülasyonda sınayabiliyoruz. Simülasyon, yüksek uyumun yanlış bir modelin sadakatle açıklanmasından da gelebileceğini
  gösterdi.
* Değerlendirme düzeyleri açısından: Makalenin uygulayıcı görüşmeleri **insan düzeyinde**, bizim tüm ölçümlerimiz **işlev
  düzeyinde**. İkisi birbirini tamamlıyor; işlev düzeyindeki bulgular, uygulayıcıların sezgiyle yaptığı seçime bir kural öneriyor.

## Sınırlılıklar ve sapmalar

* COMPAS ön işlemesi makalede raporlanmamış. ProPublica'nın standart filtrelerini ve `score_text == "High"` etiketini kullandık.
  Makalenin özellik listesine uymak için `two_year_recid`'i (sanığın gerçek sonucu) girdi olarak tuttuk; bu, risk puanını tahmin etmek
  için gerçekçi olmayan bir bilgi.
* Makalenin sayısal sonuçlarının çoğu yalnızca ısı haritalarında; tablo olarak verilmemiş. Bu yüzden karşılaştırmamız iddia düzeyinde.
* German Credit yerine kendi verimiz olarak Taiwan verisini kullandık. Bu, "kendi verinizle yeniden üretin" koşulunu karşılıyor ama
  makalenin German Credit sonuçlarıyla doğrudan sayısal karşılaştırma yapmıyor.
* LIME çekirdek genişliği, KernelSHAP arka planı, IG tabanı ve SmoothGrad gürültüsü makalede raporlanmamış. Sonuçlar bu seçimlere
  duyarlı olabilir; seçimlerimizi yukarıda açıkça yazdık.
* Sinir ağı scikit-learn ile eğitildi. Makalenin eğitim ayrıntıları (optimizasyon, dönem sayısı) raporlanmamış.
* Makalenin uygulayıcı çalışması (insan düzeyi) ile metin ve görüntü deneyleri yeniden üretilmedi.

## Dosyalar ve çalıştırma

| Dosya | İçerik |
|---|---|
| `analiz.ipynb` | Çalıştırılmış analiz defteri (tam mod) |
| `analiz.py` | Defterin aynısı, düz Python ([jupytext](https://jupytext.readthedocs.io) "percent" biçimi) |
| `sekiller/` | Defterdeki grafiklerin 200 dpi kopyaları |
| `requirements.txt` | Gerekli paketler (PyTorch gerekmez) |

```bash
pip install -r requirements.txt
jupyter notebook analiz.ipynb
```

* **Veri:** COMPAS, çalışma anında ProPublica'nın GitHub deposundan indirilir. Kişi adları içerdiği için depoya eklenmedi; kod yalnızca
  yedi özniteliği ve etiketi kullanır. Taiwan verisi, uygulama klasöründeki CSV'den okunur
  (`../../uygulama/berkaycongar-20269258007/veri/`).
* **Süre:** Tam mod (`HIZLI = False`) iki çekirdekli bir makinede yaklaşık 7 dakika sürer. Canlı gösterim için `HIZLI = True`
  (yaklaşık 1,5 dakika).

## Kaynaklar

* Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S. ve Lakkaraju, H. (2022). The disagreement problem in explainable
  machine learning: A practitioner's perspective. arXiv:2202.01602.
* Molnar, C. (2025). *Interpretable Machine Learning* (3. baskı), Bölüm 32. https://christophm.github.io/interpretable-ml-book/evaluation.html
* Doshi-Velez, F. ve Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv:1702.08608.
* Ribeiro, M. T., Singh, S. ve Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *KDD*.
* Lundberg, S. M. ve Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
* Sundararajan, M., Taly, A. ve Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
* Smilkov, D., Thorat, N., Kim, B., Viégas, F. ve Wattenberg, M. (2017). SmoothGrad: Removing noise by adding noise. arXiv:1706.03825.
* Simonyan, K., Vedaldi, A. ve Zisserman, A. (2014). Deep inside convolutional networks: Visualising image classification models and saliency maps. *ICLR Workshop*.
* Angwin, J., Larson, J., Mattu, S. ve Kirchner, L. (2016). Machine bias. *ProPublica*. Veri: https://github.com/propublica/compas-analysis
* Yeh, I.-C. ve Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default
  of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
