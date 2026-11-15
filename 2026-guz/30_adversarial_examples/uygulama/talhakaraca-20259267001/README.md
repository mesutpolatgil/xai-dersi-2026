# Bölüm 30 Uygulaması: Kötücül Örnekler Laboratuvarı

`talhakaraca-20259267001.html` dosyasını tarayıcıda açın. Kurulum ve internet gerekmez; model sayfanın içinde, tarayıcıda çalışır.

Açılış sayfasında her deney için bir kart vardır. Karta tıklayınca o deneyin sayfası açılır.

| # | Deney | Kitaptaki başlık | Ne yapılır? |
|---|-------|------------------|-------------|
| 1 | Hedefli saldırı (L-BFGS) | Köpeğimde Bir Terslik Var | Hedef sınıf ve c seçilir; her giysi o sınıfa dönüştürülür |
| 2 | FGSM | Rahatsız Edilmiş Panda | ε kaydırılır; aynı büyüklükte rastgele gürültüyle karşılaştırılır |
| 3 | 1-piksel saldırısı | Bir Denizanası… Hayır, Bekle. Bir Küvet | Diferansiyel evrim nesil nesil izlenir |
| 4 | Adversarial yama | Her Şey Bir Tost Makinesi | 7×7 yama görüntünün üzerinde sürüklenir |
| 5 | Sağlam örnekler (EOT) | 3D Baskılı Kaplumbağa | Nesne kaydırılır; normal saldırı ile EOT karşılaştırılır |
| 6 | Kara kutu saldırısı | Gözü Bağlı Düşman | Rastgele orman sorgulanır, vekil eğitilir, örnekler aktarılır |
| 7 | Savunma | Siber Güvenlik Perspektifi | Normal model ile adversarial eğitimli model karşılaştırılır |

## Modeller

- **Veri:** Fashion-MNIST (28×28, 10 kategori).
- **Hedef model:** 784 → 128 → 10 sinir ağı, test doğruluğu %87.
- **Adversarial eğitimli model:** aynı mimari, eğitimde FGSM örnekleri (ε = 0,1) eklenir, test doğruluğu %80.
- **Kara kutu:** 20 ağaçlı rastgele orman.
- **Yama deneyi:** küçük bir evrişimli ağ (conv 5×5 → global max → 64 → 10), test doğruluğu %79. 7×7 yama, 2.000 test görüntüsünde görüntülerin %94'ünü "Çanta" olarak sınıflandırtır; aynı boyutta rastgele yama %1,5.

## Basitleştirmeler

- Deney 1'de kitaptaki L-BFGS yerine aynı amaç fonksiyonu gradyan inişiyle çözülür.
- Deney 5'te dönüşüm olarak yalnızca ±2 piksel kaydırma kullanılır (25 konum).
- Deney 4 ve 5, kitaptaki fiziksel saldırıların küçük ölçekli benzerleridir.

## Yeniden üretme

`kaynak/` klasöründe eğitim ve derleme kodu ile eğitilmiş model dosyaları vardır. Fashion-MNIST verisi `kaynak/veri/` klasöründen okunur; klasör yoksa veri ilk çalıştırmada indirilir (yaklaşık 30 MB). Bu klasör `.gitignore` ile depoya eklenmez.

```bash
cd kaynak
pip install numpy scikit-learn pillow
python egit.py --hepsi          # modelleri ve yamayı yeniden eğitir (yaklaşık 6 dakika)
python build.py ../talhakaraca-20259267001.html
```
