# Interpretation of Neural Networks is Fragile

**Künye.** Amirata Ghorbani, Abubakar Abid, James Zou. *Interpretation of Neural Networks is Fragile.* AAAI 2019. arXiv:1710.10547. Orijinal kod: `github.com/amiratag/InterpretationFragility`.

**Kitap bölümü:** Molnar, *Interpretable Machine Learning* — Bölüm 28, Saliency Maps (Pixel Attribution).
**Makale tipi:** Seçenek 1 (Zafiyet / Manipülasyon).

## 1. Neden bu makale?

Molnar'ın bölümü, saliency haritalarının "modelin neye baktığını" gösterdiğini anlatır ve sonunda bu yöntemlerin güvenilirliğine dair uyarılar ekler. Bu makale o uyarının doğrudan deneysel karşılığıdır: **modeli değiştirmeden, girdiye çok küçük bir gürültü ekleyerek açıklamayı bozmak**. Dersin "bozarak anlama" (red-teaming) felsefesine tam oturur.

## 2. Problem

Bir sınıflandırıcı $f$ ve bir girdi $x$ için bir yorumlama yöntemi $\mathcal{I}(x; f)$ (burada: piksel önem haritası) üretilir. İyi bir yorumlama, benzer girdiler için benzer olmalıdır. Makale sorar: **tahmini değiştirmeyen, gözle ayırt edilemeyen bir $\delta$ ile $\mathcal{I}(x+\delta)$ ile $\mathcal{I}(x)$ çok farklı hâle getirilebilir mi?**

## 3. Yöntem

İncelenen yöntemler: Simple Gradient (saliency), DeepLIFT ve Integrated Gradients. Önem haritası piksel başına tek bir sayıya indirgenir.

**Saldırı (iteratif).** $\|\delta\|_\infty \le \varepsilon$ ve $\arg\max f(x+\delta) = \arg\max f(x)$ kısıtlarıyla, haritayı bozan bir kayıp azaltılır. Makale üç saldırı tanımlar:

* **Top-k saldırısı:** Orijinal haritadaki ilk $k$ pikselin toplam önemini düşürür.
* **Ağırlık merkezi (mass-center) saldırısı:** Haritanın ağırlık merkezini mümkün olduğunca uzağa kaydırır.
* **Hedefli saldırı:** Önemi belirli bir bölgeye taşır.

Bu çalışmadaki yeniden üretim **top-k saldırısını** kapsar:

$$\mathcal{L}(x+\delta)=\frac{\sum_{(i,j)\in T_k(x)} S(x+\delta)_{ij}}{\sum_{(i,j)} S(x+\delta)_{ij}},\qquad \delta \leftarrow \delta-\alpha\,\mathrm{sign}(\nabla_\delta \mathcal{L}).$$

**Ölçütler.** Orijinal ve bozulmuş harita arasında (i) top-$k$ kesişimi, (ii) Spearman rank korelasyonu, (iii) ağırlık merkezi kayması.

**Kontrol deneyi.** Aynı büyüklükte *rastgele* işaretli gürültü, haritayı çok az bozar. Bu, bozulmanın gürültünün büyüklüğünden değil **yönünden** geldiğini gösterir.

## 4. Bulgular (makalenin nitel sonucu)

* Küçük, sistematik gürültüyle etiket değişmeden harita belirgin biçimde değişir.
* Rastgele gürültü haritaları çok daha az değiştirir.
* Bulgular birden fazla yöntemde ve hem ImageNet hem CIFAR-10 modellerinde görülür.
* Makale sonuçları örnek tabanlı yorumlamalara (influence functions) da genişletir.

## 5. Eleştiri ve sınırlar

* Saldırı gradyanı gerektirir (beyaz kutu); siyah kutu senaryosu ayrı bir soru.
* ReLU'nun ikinci türevi sıfır olduğu için saldırı gradyanı yaklaşıklanır. Bu, sonuçların uygulama ayrıntılarına duyarlı olmasına yol açabilir.
* Dombrowski ve ark. (2019, arXiv:1906.07983) sorunun kaynağını karar yüzeyinin geometrisine bağlar ve yumuşak aktivasyonlarla (softplus) hafifletilebildiğini gösterir.
* "Harita değişti" ifadesi "açıklama yanlış" demek değildir; hangi haritanın doğru olduğunu söyleyecek bir referans yoktur. Bu yüzden Adebayo ve ark. (2018, *Sanity Checks for Saliency Maps*) ile birlikte okunmalıdır.

## 6. Bu depodaki yeniden üretim (`analiz.ipynb`)

| Konu | Makale | Bu çalışma |
|---|---|---|
| Model | SqueezeNet, CIFAR-10 CNN | ImageNet ön eğitimli ResNet-18 |
| Veri | ImageNet, CIFAR-10 | Imagenette (ImageNet alt kümesi) |
| Yöntemler | Saliency, DeepLIFT, IG | Simple Gradient, IG (+ SmoothGrad savunma analizi) |
| Saldırılar | Top-k, mass-center, hedefli | Top-k |
| Yeni katkı | | ε duyarlılığı, SmoothGrad'in aktarılabilirlik testi |

Sonuçlar makaledeki sayılarla birebir aynı olmak zorunda değildir; model, veri, ε ve iterasyon sayısı farklıdır. Defterin son bölümü gerçek sayıları otomatik raporlar.

## 7. Kaynaklar

* Ghorbani, A., Abid, A., Zou, J. (2019). Interpretation of Neural Networks is Fragile. AAAI. arXiv:1710.10547.
* Dombrowski, A.-K. ve ark. (2019). Explanations can be manipulated and geometry is to blame. NeurIPS. arXiv:1906.07983.
* Adebayo, J. ve ark. (2018). Sanity Checks for Saliency Maps. NeurIPS. arXiv:1810.03292.
* Smilkov, D. ve ark. (2017). SmoothGrad: removing noise by adding noise. arXiv:1706.03825.
* Sundararajan, M. ve ark. (2017). Axiomatic Attribution for Deep Networks. ICML. arXiv:1703.01365.
* Molnar, C. *Interpretable Machine Learning*, Bölüm 28.
