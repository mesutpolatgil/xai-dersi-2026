# Bölüm 2 · Önem Haritaları İçin Sağlama Testleri

Bu klasör, yüksek lisans projesinin ikinci bölümü için hazırlanan akademik Türkçe çeviriyi ve ilgili destek dosyalarını içerir.

- **Türkçe çeviren:** Tania El Hawari
- **Öğrenci numarası:** 20259258043
- **Özgün makale:** *Sanity Checks for Saliency Maps*
- **Özgün yazarlar:** Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt ve Been Kim
- **Yayın:** NeurIPS 2018, *Advances in Neural Information Processing Systems*, cilt 31. [Yayın kaydı](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)
- **Çeviriye esas sürüm:** 6 Kasım 2020 tarihli [arXiv:1810.03292v3](https://arxiv.org/abs/1810.03292v3) · [PDF](https://arxiv.org/pdf/1810.03292v3) · [HTML](https://arxiv.org/html/1810.03292v3)

## Dosya yapısı

```text
ceviri/
├── .gitignore
├── README.md
├── makale.qmd
├── references.bib
└── images/
    ├── figure-01.png
    ├── …
    └── figure-37.png
```

- `makale.qmd` çevirinin Quarto kaynak belgesidir. Kaynakça metnini ve Base64 kodlu PNG şekillerini kendi içinde barındırır.
- `references.bib` özgün makalenin 43 kaynak kaydını (`ref01`–`ref43`) ve makalenin ayrı künyesini (`adebayo2018sanity`) içerir.
- `images/` kaynak belgeye gömülü 37 PNG’nin bayt düzeyinde eş kopyalarını içerir. Dosya numaraları çevirideki şekil numaralarını izler.
- `.gitignore` Quarto’nun yerel çalışma dosyalarını, önbellekleri ve yaygın işletim sistemi artıklarını kapsar.

Mevcut QMD, kaynakçayı ve şekilleri doğrudan kendi içinden okur. Ayrı BibTeX ve PNG dosyaları başvuru amacıyla hazırlanmıştır.

## Çeviri dosyasının kullanımı

Çeviri, VS Code’da `makale.qmd` dosyası açılarak okunabilir ve düzenlenebilir. Bu klasör Quarto kaynak belgesini ve destek dosyalarını içerir. HTML çıktısı teslim paketine dahil değildir.

## Kaynakça ve telif

BibTeX kayıtları QMD’deki kaynakçadan çıkarıldı ve arXiv, yayıncı veya konferans kayıtlarından doğrulanan yayın bilgileriyle tamamlandı. Her kayıt için bağımsız yayıncı doğrulaması yapılmadı. Doğrulanmamış DOI veya diğer eksik alanlar eklenmedi. `ref32`, özgün kaynakçada anılan *Automatic Shadow Detection in 2D Ultrasound* sürümüne bağlanır. Daha sonra yayımlanan *Automatic Shadow Detection in 2D Ultrasound Images* sürümünün künyesi farklıdır.

arXiv kaydındaki münhasır olmayan lisans, arXiv’e makaleyi dağıtma yetkisi verir. Bu lisans, tam Türkçe çeviriyi veya özgün şekilleri GitHub gibi kamuya açık bir ortamda yeniden yayımlamak için üçüncü kişilere izin vermez. Kamuya açık paylaşım öncesinde hak sahibinin izni veya uygulanabilir başka bir hukuki dayanak doğrulanmalıdır. Bu husus, görsellerin gömülü olduğu QMD ile ayrı PNG dosyaları için de geçerlidir. [arXiv lisans metni](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
