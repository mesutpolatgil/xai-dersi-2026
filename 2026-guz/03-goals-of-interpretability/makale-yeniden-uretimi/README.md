# Model Açıklamaları İçin Hata Ayıklama Testleri — Makale Yeniden Üretimi

Bu klasör, **Julius Adebayo, Michael Muelly, Ilaria Liccardi ve Been Kim'in _Debugging Tests for Model Explanations_ (NeurIPS 2020; arXiv:2011.05429v1)** makalesindeki üç bozulma kategorisine ait dört hata örneğini özgün **sentetik** küçük veride ele alan eğitimsel yeniden uygulamayı içerir.

**Önemli:** Bu çalışma, özgün kuş–köpek fotoğraflarını, ImageNet/Fashion-MNIST/MNIST çapraz-modellerini, yayınlanan sayısal sonuçları veya **54 katılımcılı insan çalışmasını** tekrar etmez. Notebook'ta kaydedilen sonuçlar yalnızca burada yürütülen deneylere aittir. Gerçek veri ile canlı uygulama isteniyorsa bu ayrı bir adımdır; tamamlandı diye iddia edilmez.

## Seçenek 1 — Zafiyet/Manipülasyon Analizi

Ödevde Seçenek 1 seçilmiştir. *Debugging Tests for Model Explanations*, açıklamaların model kusurlarını her koşulda ortaya çıkarıp çıkaramadığını sorgular. Bu projede sahte arka plan, yanlış eğitim etiketi, kademeli ağırlık yeniden başlatma ve dağılım dışı girdi vekili, açıklama haritalarının farklı bozulmalara duyarlılığını sınar. Deneyler küçük sentetik veriyle yürütülen kavramsal uyarlamalardır. Ters arka plan örneğinde doğruluk düşmemiştir.

## Dosyalar

| Dosya / klasör | Açıklama |
|---|---|
| `analiz.ipynb` | Üretim, eğitim, yedi açıklama yöntemi, dört kontrollü test, rapor oluşturma |
| `veri.csv` | 1056 PNG'nin yolu, sınıfı, bölmesi, koşulu, tohumu ve SHA-256 özeti |
| `data/` | Sentetik örnekler ve kontrol koşulları: `synthetic/`, `spurious/`, `counterfactual/`, `background_only/`, `ood/` |
| `sekiller/` | 10 orijinal deney grafiği (kaynak makalenin şekillerinin kopyası değil) |
| `model_scores.csv` | Gerçek koşudan hesaplanan doğruluk ve kayıplar |
| `background_attribution.csv` | Dış çerçeve için betimleyici atıf oranı |
| `benzerlik_sonuclari.csv` | Yanlış etiket, ağırlık ve OOD karşılaştırmaları |
| `makale.md`, `makale.html` | Notebook sonuçlarından oluşturulan Türkçe rapor / tarayıcı çıktısı |
| `requirements.txt` | Kullanılacak Python paketleri |
| `verify_project.py` | Veri/rapor/görsel bağlantıları için yerel bütünlük denetimi |
| `.gitignore` | Ortam, geçici ve ham kişisel veri dışlama kuralları |

## Kurulum ve çalıştırma

Komutların çalışma dizini `2026_guz/03_goals-of-interpretability/makale-yeniden-uretimi/` klasörüdür.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

VS Code'da **analiz.ipynb** açın, `.venv` Python çekirdeğini seçin, **Restart & Run All** uygulayın. Defterin son hücresi `makale.md` ve `makale.html` dosyalarını yeniden oluşturur. Her koşuda aynı tohum kullanılır. CPU veya kütüphane sürümü farklılıkları küçük sapmalara yol açabilir.

Kontrol için:

```powershell
.\.venv\Scripts\python.exe verify_project.py
```

Denetim betiği, dosya özetlerinin yanı sıra piksel ve tohum düzeyinde bölme ayrımını, ölçüm kayıtlarını, rapor tablolarını ve Markdown ile HTML eşleşmesini kontrol eder. Aynı bölme içindeki 217 birebir kopya, aynı tohum ve arka planla üretilen eşleştirilmiş veri koşullarından kaynaklanır. Bunlar bağımsız örnek olarak yorumlanmaz.

Raporun `sekiller/` bağlantılarının düzgün çalışması için `makale.html` dosyasını aynı dizin düzeninde tutun. MathJax denklemleri tarayıcıda çevrimiçi yüklenir.

## Bilimsel sınırlar

- Özgün makalede veri, model ve test zamanı olmak üzere üç bozulma kategorisi, dört hata örneği ve 54 kişilik bir insan çalışması bulunmaktadır. İnsan katılımcı veya gerçek fotoğraf çalışması **yapılmadı**.
- Üretilen görseller gerçek hayvanları değil **basitleştirilmiş simgeleri** gösterir. Kaynak makalenin fotoğrafları ya da sonuçları gibi sunulmamalıdır.
- Test zamanı OOD kontrolü sadece renk tersleme/dönüş/doku değişimi uygular; orijinal makaledeki **çapraz veri-alanı** deneyi değildir.
- Sahte ipucuyla eğitilen model, uyumlu ve ters arka plan testlerinde 96 örneğin tamamını doğru sınıflandırdı ve yalnız arka plan testinde %50 doğruluk elde etti. Ters arka planda çapraz entropi yükseldi. Bu sonuç arka planın skorları etkileyebildiğini gösterir, ancak sınıf kararlarında baskın bir ipucu olduğunu göstermez. Dış çerçeve atıf oranı özgün GT-1/GT-2 maskelerinin karşılığı değildir.
- Sahte arka plan, yanlış etiket, ağırlık yeniden başlatma ve OOD deneyleri tek tohum ve kısıtlı veriyle yürütülür. Ölçümlerin genellenebilirliği sınırlıdır.
- Fotoğraf ve gerçek veri gereklilikleri varsa öğretmenin resmi yönergesi üzerinden ayrıca tamamlanmalıdır.

## AI desteği ve akademik şeffaflık

Notebook, rapor, deney tasarımı ve doğrulama betiği AI yardımıyla hazırlanmıştır. Yapı ve ölçümler kod çalıştırmasıyla denetlenir; AI'nin yazmış olduğu bir cümle, bağımsız bilimsel doğrulamanın yerini tutmaz.

Özgün kaynak: <https://arxiv.org/pdf/2011.05429v1>.
