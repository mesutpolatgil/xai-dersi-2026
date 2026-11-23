# Bölüm 3 · Model Açıklamaları İçin Hata Ayıklama Testleri

Bu klasör, *Goals of Interpretability* (Yorumlanabilirliğin Amaçları) kapsamında seçilen **Debugging Tests for Model Explanations** araştırma makalesine ilişkin akademik Türkçe çalışma metnini, kaynak PDF'den elde edilen şekilleri ve destek dosyalarını içerir. Molnar'ın kitap bölümünün kendisi değildir.

- **Türkçe hazırlayan:** Tania El Hawari
- **Öğrenci numarası:** 20259258043
- **Özgün makalenin yazarları:** Julius Adebayo, Michael Muelly, Ilaria Liccardi, Been Kim
- **Yayın:** NeurIPS 2020
- **Çeviriye esas kaynak:** [arXiv:2011.05429v1](https://arxiv.org/abs/2011.05429v1), 10 Kasım 2020 — [HTML](https://arxiv.org/html/2011.05429v1) · [PDF](https://arxiv.org/pdf/2011.05429v1)

## Dosya yapısı

```text
ceviri/
├── .gitignore
├── README.md
├── makale.qmd
├── references.bib
├── original_reference_records.json
└── images/
    ├── figure-01.png
    ├── …
    ├── figure-64.png
    └── manifest.json
```

- `makale.qmd`: akademik Türkçe çalışma metninin Quarto kaynak dosyası, araştırmanın ana bölümleri, ekleri, dört numaralı tablo, Ek H’deki eşleştirme tablosu, denklemleri ve şekil bağlantıları.
- `references.bib`: PDF'de numaralandırılmadan listelenen **73 bibliyografik kayıt** ve incelenen yayının ayrı künyesi. Kaynak metni değiştirmemek için 73 kaydın tam bibliyografik dizesi `note` alanında saklanmıştır; bunlar tek tek yayıncı üzerinden doğrulanmış normalleştirilmiş BibTeX kayıtları değildir.
- `original_reference_records.json`: PDF sayfa 13–18'den çıkarılan kaynakça kayıtlarının kontrol amaçlı metin kopyası.
- `images/`: **64** kaynak şekil, özgün PDF sayfalarından kırpılmıştır. Şekil numarası özgün makaledeki sayıya karşılık gelir. PNG içindeki İngilizce yazılar kaynak görsellerin bir parçasıdır.
- `images/manifest.json`: her PNG'nin özgün PDF sayfası, kırpma koordinatları ve **SHA-256** özeti.
- `.gitignore`: Quarto'nun ve işletim sisteminin geçici dosyalarını yok sayar; QMD, PDF'den çıkarılmış görseller ve kaynakça tutulur.

## Metnin kapsamı ve doğrulama durumu

Metin Özet, 1–6. bölümler ve A–I eklerini, dört numaralı kaynak tabloyu ve Ek H’deki eşleştirme tablosunu içerir. Kaynakça, özgün PDF’deki gibi eklerden önce yer alır. Son denetimde 2–6. bölümlerin seçili paragrafları ve Ek A, B ve H’de eksik bulunan ayrıntılar genişletildi. Ek D.1, D.2, E ve F’nin dört kaynak paragrafı PDF’nin 24–25. sayfalarıyla tam metin düzeyinde karşılaştırıldı. Şekil açıklamaları kaynak metinle karşılaştırıldı ve 64 açıklamanın numara sırası korundu. Şekillerin içindeki eksen ve arayüz yazıları özgün görsellerin parçasıdır.

Tablo 2 ve 3’ün toplam 240 sayısal hücresi kaynak PDF ile karşılaştırıldı. Özgün kaynakçadaki 73 kayıt PDF’nin 13–18. sayfalarıyla eşleşir. BibTeX’te kayıtlar ağırlıklı olarak tam kaynak dizelerini içeren `note` alanlarıdır. DOI, cilt, sayı, sayfa aralığı ve yayınevi gibi alanların tamamı bağımsız yayıncı kayıtlarıyla doğrulanmamıştır. Kaynakta bulunmayan bilgiler eklenmemiştir.

Belgenin bütünü için her paragrafı ve matematiksel gösterimi kapsayan bir eksiksizlik denetimi henüz tamamlanmamıştır. Son kontrolde kaynak başlıkları, matematik ayraçları, görsel yolları ve atıf hedefleri denetlendi. Quarto CLI bu ortamda bulunmadığından Quarto derleme doğrulaması yapılmadı ve HTML üretilmedi.

**Özgün PDF'nin kendi içindeki uyuşmazlık:** Ana metin, yanlış etiket testinin eğitim doğruluğunu **%93,2** (PDF s. 8), Ek D.2 ise **%94,2** (PDF s. 24) olarak verir. İki değer değiştirilmeden ve kaynak konumları belirtilerek aktarılmıştır. Bu uyuşmazlık çeviriden kaynaklanmaz. Ayrıca PDF ilk sayfada ikinci yazarı *Micheal Muelly* olarak yazar; arXiv yayın kaydındaki biçim *Michael Muelly* olduğundan künyede bu biçim kullanılmıştır.

**Kaynak şekilleri:** PDF sayfasından kırpılan PNG'ler, arXiv HTML'deki orijinal resim dosyalarının bayt düzeyinde kopyası değildir. Bütün PNG dosyalarının SHA-256 değerleri manifestoyla karşılaştırılmış ve 64 kırpma bölgesi yerel PDF üzerinde yeniden görüntülenmiştir. Şekil 11’in sol kenarı, komşu sütun metnini çıkarmak için daraltılmıştır. Bazı kırpmalarda bitişik sayfa öğeleri görülebilir. Dışarıdan alınmış/generatif yapay görsel kullanılmamıştır. Şekillerin tamamı için `images/manifest.json` ile kaynak PDF izlenebilir.

## Çeviri dosyasının kullanımı

Bu klasörde yalnızca kaynak `.qmd` belgesi ve onu destekleyen kaynakça ile görseller bulunmaktadır; **`makale.html` kasıtlı olarak pakete eklenmemiştir.** Metni VS Code'da `makale.qmd` üzerinden okuyabilir veya düzenleyebilirsiniz. `.qmd` dosyası, Quarto Markdown biçimidir. `references.bib` ve `images/` klasörünü aynı dizinde tutun. Quarto yapılandırmasında HTML çıktı ayarları bulunsa da, teslim paketi HTML içermez.

## Telif, lisans ve akademik şeffaflık

Özgün arXiv çalışması **Creative Commons Attribution 4.0 International (CC BY 4.0)** koşullarıyla listelenmektedir: <https://creativecommons.org/licenses/by/4.0/>. Yazarlar, özgün kaynak ve lisans açıkça belirtilmiştir; bu çalışma Türkçeye çeviri/uyarlama ve PDF'den görsel kırpma içerir. Özgün sayfalardaki yazar, model, deney, yöntem ve sonuçlar incelenen araştırmaya aittir. Bu klasörde yeni deneylerin yürütüldüğü ya da sonuçların bağımsız olarak doğrulandığı iddia edilmez.

**Not:** Bu klasör, `makale-yeniden-uretim/` ve `sunum/` çalışmalarından ayrı tutulmalıdır. GitHub'a yüklemeden önce akademik içerik ve uygun atıf, öğretim elemanının ödev yönergesine göre tekrar gözden geçirilmelidir.
