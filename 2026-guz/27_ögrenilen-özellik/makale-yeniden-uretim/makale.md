# Don't Trust Your Eyes: On the (Un)reliability of Feature Visualizations

**Künye.** Robert Geirhos, Roland S. Zimmermann, Blair Bilodeau, Wieland Brendel, Been Kim. *Don't trust your eyes: on the (un)reliability of feature visualizations.* ICML 2024. arXiv:2306.04719.

**Kitap bölümü:** Molnar, *Interpretable Machine Learning* — Bölüm 27, Learned Features (Feature Visualization, Network Dissection).
**Makale tipi:** Seçenek 1 (Zafiyet / Manipülasyon).

## 1. Neden bu makale?

Molnar'ın bölümü, feature visualization'ın (FV) ağın öğrendiği özellikleri gösterdiğini ve "yorumlanabilirlik yanılsaması" riskini anlatır. Bu makale yanılsama riskini **resmî olarak ve deneysel olarak** ele alır: FV'nin doğal veri üzerindeki davranışı yansıtıp yansıtmadığı garanti değildir, hatta kasıtlı olarak bozulabilir.

## 2. Problem

Bir birim (nöron/kanal) $h_z$ için FV, aktivasyonu maksimize eden girdiyi arar:

$$\mathbf{x}^*=\arg\max_{\mathbf{x}}\; h_z(\mathbf{x}).$$

Bu $\mathbf{x}^*$ genelde doğal görüntü dağılımının dışındadır. Soru: $\mathbf{x}^*$, birimin **doğal verideki davranışı** hakkında güvenilir bilgi verir mi?

## 3. Yöntem (makalenin ana fikirleri)

1. **Ağ manipülasyonu (fooling circuit).** Ağa bir devre eklenir; bu devre FV optimizasyonunun bulduğu türden dağılım dışı girdileri tespit eder ve birimin aktivasyonunu değiştirir. Doğal girdilerde devre tetiklenmez, ağın doğal davranışı aynı kalır. Sonuç: FV istenen (ör. anlamsız ya da yanıltıcı) görüntüyü gösterir.
2. **Standart ağlarda güvenilirlik analizi.** Manipüle edilmemiş ağlarda bile FV'nin doğal veri davranışını ne kadar öngördüğü incelenir.
3. **Teorik çerçeve.** Bir FV'nin neyi garanti edip etmediği hakkında biçimsel bir tartışma.

## 4. Bulgular (nitel)

* FV, doğal davranışı koruyan bir ağ manipülasyonuyla keyfi biçimde değiştirilebilir.
* FV'ye bakmak, birimin doğal veride ne yaptığını doğrulamak için yeterli değildir.
* Bu yüzden FV, doğal veri örnekleri ve müdahale deneyleriyle birlikte kullanılmalıdır.

## 5. Eleştiri ve sınırlar

* Manipülasyon, ağ üzerinde kontrol sahibi bir saldırgan varsayar; "ağ olduğu gibi eğitildiğinde" FV'nin ne kadar güvenilir olduğu ayrı bir sorudur.
* Tek bir FV yöntemi (aktivasyon maksimizasyonu) ve belirli mimariler incelenir; farklı düzenlileştirmeler farklı sonuç verebilir.
* "FV güvenilmez" sonucu, FV'nin işe yaramaz olduğu anlamına gelmez; yalnızca tek başına kanıt olmadığını söyler.

## 6. Bu depodaki yeniden üretim (`analiz.ipynb`)

**Önemli not.** Makalenin devresi birebir kopyalanmadı. Aynı mantığın sadeleştirilmiş hâli uygulandı: girdinin yüksek frekans enerjisi doğal görüntülerde görülmeyen eşiği aşınca devreye giren bir kapı, seçili kanala yüksek bir terim ekler.

| Konu | Makale | Bu çalışma |
|---|---|---|
| Model | Inception V1 vb. | GoogLeNet (Inception V1), `inception4d` |
| Veri | ImageNet | Imagenette (ImageNet alt kümesi) |
| Manipülasyon | Özel tasarlanmış devre | Sadeleştirilmiş HF-enerji kapısı |
| FV | Lucid tarzı | Piksel uzayında Adam + jitter + TV |
| Ölçüm | Doğal davranış, FV | Doğal skor/logit farkı, FV tepkisi, HF enerjisi |

## 7. Network Dissection ile bağlantı

Network Dissection (Bau ve ark. 2017) birimleri piksel etiketli doğal görüntülerle (Broden) hizalar. FV'den farklı olarak doğal veri üzerinde ölçülür, bu yüzden bu tür bir manipülasyona daha az açıktır. Ancak piksel etiketli veri gerektirir.

## 8. Kaynaklar

* Geirhos, R. ve ark. (2024). Don't trust your eyes: on the (un)reliability of feature visualizations. ICML. arXiv:2306.04719.
* Olah, C., Mordvintsev, A., Schubert, L. (2017). Feature Visualization. Distill.
* Bau, D. ve ark. (2017). Network Dissection: Quantifying Interpretability of Deep Visual Representations. CVPR. arXiv:1704.05796.
* Molnar, C. *Interpretable Machine Learning*, Bölüm 27.
