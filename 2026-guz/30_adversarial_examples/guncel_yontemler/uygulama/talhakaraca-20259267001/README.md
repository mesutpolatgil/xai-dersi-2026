# Embedding Lab: Görüntü modellerinde embedding saldırıları ve savunmaları

`talhakaraca-20259267001.html` tarayıcıda açılan etkileşimli uygulamadır. İnternet bağlantısı gerektirmez. Sonuçlar önceden hesaplandı.

`kaynak/` klasörü bu sonuçları üreten kodu içerir. Bütün komutlar `kaynak/` klasöründe çalıştırılır. `build.py` sayfayı `kaynak/embedding-lab.html` olarak üretir; üst klasördeki `talhakaraca-20259267001.html` bu dosyanın kopyasıdır.

Veri (`veri/`), ara sonuçlar (`cikti/`) ve şekiller (`sekil/`) repoya eklenmez. Kod bunları kendisi indirir veya üretir.

## Modeller ve veri

| Ne | Kaynak |
|---|---|
| CLIP ViT-B/32 | OpenCLIP, `laion2b_s34b_b79k` |
| FARE (gürbüz CLIP) | `hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K` |
| TeCoA (gürbüz CLIP) | `hf-hub:chs20/TeCoA4-ViT-B-32-laion2B-s34B-b79K` |
| Transfer modeli | OpenCLIP `ViT-B-16-quickgelu`, `openai` |
| Difüzyon modeli | `google/ddpm-cifar10-32` (diffusers) |
| Imagenette | https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz (yalnızca `val`) |
| CIFAR-10 test | https://huggingface.co/datasets/uoft-cs/cifar10 (`plain_text/test-00000-of-00001.parquet`) |

Model ağırlıkları ilk çalıştırmada kendiliğinden iner.

## Kurulum

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open_clip_torch diffusers huggingface_hub safetensors scikit-learn pandas pyarrow matplotlib pillow
```

Veriyi `kaynak/veri/` klasörüne koyun:

```bash
cd kaynak
mkdir -p veri && cd veri
curl -L -o imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz imagenette2-320/val && rm imagenette2-320.tgz
curl -L -o cifar_test.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
cd ..
```

## Çalıştırma sırası

| Betik | Ne yapar | Süre (2 çekirdek CPU) |
|---|---|---|
| `d1_embedding.py` | Embedding'ler, benzerlik, arama, sözlük | ~1 dk |
| `d3_tipografik.py` | Tipografik saldırı, head ablation | ~9 dk |
| `d3b_bos.py` | Yazısız kutu kontrolü (demo) | <1 dk |
| `d4_pgd.py` | PGD: hedefsiz, hedefli ("muz"), görüntüden görüntüye, transfer | ~50 dk |
| `d6_harita.py` | Ortak t-SNE haritası, alaka haritaları | ~2 dk |
| `d5_difuzyon.py` | Difüzyon: ileri süreç, üretim, DiffPure | ~15 dk |
| `d2_transformer.py 10 255` | Tek yama saldırısı | ~8 dk |
| `build.py` | `embedding-lab.html` dosyasını üretir | <1 dk |
| `fig_sunum.py` | Sunum şekillerini `sekil/` klasörüne çizer | <1 dk |

GPU varsa süreler çok kısalır (kod CPU için yazıldı, `torch.set_num_threads` kullanır).

## Dosyalar

- `ortak.py`: model yükleme, sıfır-atış sınıflandırıcı, PGD
- `vit_araclar.py`: ViT attention'ını kaydeden ve head çıktısını değiştiren kanca, alaka haritası (Chefer ve ark. 2021)
- `sablon.html`: etkileşimli sayfanın şablonu
