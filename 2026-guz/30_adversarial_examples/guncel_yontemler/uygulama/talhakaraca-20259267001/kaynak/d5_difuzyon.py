"""Deney 5: Difüzyon. İleri süreç (gürültü ekleme), üretim ve DiffPure tarzı temizleme.
Veri: CIFAR-10 (32x32). Sınıflandırıcı: CLIP sıfır-atış (görüntü 224'e büyütülür)."""
import os, io, json, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler
import ortak

C = ortak.CIKTI
SINIF_EN = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SINIF_TR = ['uçak', 'otomobil', 'kuş', 'kedi', 'geyik', 'köpek', 'kurbağa', 'at', 'gemi', 'kamyon']
TSTAR = 100
torch.manual_seed(0)

df = pd.read_parquet(f'{ortak.VERI}/cifar_test.parquet')
sec = []
for c in range(10):
    sec += list(df.index[df['label'] == c][:10])
X = torch.stack([torch.from_numpy(np.asarray(Image.open(io.BytesIO(df.loc[i, 'img']['bytes'])).convert('RGB'),
                                              dtype=np.float32) / 255).permute(2, 0, 1) for i in sec])
Y = torch.tensor([int(df.loc[i, 'label']) for i in sec])

m = ortak.model_yukle('clip')
T = ortak.metin_emb(m, [ortak.SABLON.format(c) for c in SINIF_EN])
def f(x32):
    return ortak.logits(m, F.interpolate(x32, size=224, mode='bicubic', align_corners=False).clamp(0, 1), T)
def dogruluk(x):
    with torch.no_grad():
        return (torch.cat([f(x[i:i + 25]) for i in range(0, len(x), 25)]).argmax(1) == Y).float().mean().item()

pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32')
unet = pipe.unet.eval()
sch = DDPMScheduler.from_config(pipe.scheduler.config)
ab = sch.alphas_cumprod


def gurultu_ekle(x, t, z=None):
    z = torch.randn_like(x) if z is None else z
    return ab[t].sqrt() * (2 * x - 1) + (1 - ab[t]).sqrt() * z


@torch.no_grad()
def temizle(x, t=TSTAR):
    xt = gurultu_ekle(x, t)
    for s in range(t, -1, -1):
        eps = unet(xt, s).sample
        xt = sch.step(eps, s, xt).prev_sample
    return ((xt + 1) / 2).clamp(0, 1)


sonuc = {'tstar': TSTAR, 'siniflar': SINIF_TR}
sonuc['temiz'] = dogruluk(X)
Xa = torch.cat([ortak.pgd(lambda z: F.cross_entropy(f(z), Y[i:i + 25]), X[i:i + 25], 8 / 255, 10) for i in range(0, 100, 25)])
sonuc['saldiri'] = dogruluk(Xa)
print(sonuc, flush=True)
Xp = torch.cat([temizle(X[i:i + 50]) for i in range(0, 100, 50)])
sonuc['temiz_temizlenmis'] = dogruluk(Xp)
Xap = torch.cat([temizle(Xa[i:i + 50]) for i in range(0, 100, 50)])
sonuc['saldiri_temizlenmis'] = dogruluk(Xap)
print(sonuc, flush=True)

os.makedirs(f'{C}/dif', exist_ok=True)
def kaydet(x, ad):
    ortak.t2pil(x).resize((128, 128), Image.NEAREST).save(f'{C}/dif/{ad}.png')
demo = [k * 10 for k in range(10)]
sonuc['demo'] = []
for j, i in enumerate(demo):
    kaydet(X[i], f'temiz{j}'); kaydet(Xa[i], f'saldiri{j}'); kaydet(Xap[i], f'saldiri_temiz{j}')
    kaydet(((Xa[i] - X[i]) / (16 / 255) + 0.5), f'fark{j}')
    with torch.no_grad():
        p = [f(z[i:i + 1]).softmax(-1)[0] for z in (X, Xa, Xap)]
    sonuc['demo'].append({'y': int(Y[i]), 'p_temiz': [round(float(v), 4) for v in p[0]],
                          'p_saldiri': [round(float(v), 4) for v in p[1]], 'p_temizlenmis': [round(float(v), 4) for v in p[2]]})
# İleri süreç şeridi
z = torch.randn(1, 3, 32, 32)
sonuc['ileri_t'] = [0, 50, 100, 250, 500, 999]
for t in sonuc['ileri_t']:
    xt = ((gurultu_ekle(X[0:1], t, z) + 1) / 2).clamp(0, 1)[0]
    kaydet(xt, f'ileri_{t}')
xt = ((gurultu_ekle(Xa[0:1], TSTAR) + 1) / 2).clamp(0, 1)[0]
kaydet(xt, 'saldiri_gurultulu0')
# Saf gürültüden üretim (DDIM, 50 adım), ara adımlar
ddim = DDIMScheduler.from_config(pipe.scheduler.config); ddim.set_timesteps(50)
xt = torch.randn(8, 3, 32, 32)
with torch.no_grad():
    for k, s in enumerate(ddim.timesteps):
        xt = ddim.step(unet(xt, s).sample, s, xt).prev_sample
        if k in (0, 10, 25, 40):
            for b in range(8):
                kaydet(((xt[b] + 1) / 2).clamp(0, 1), f'uretim_{k}_{b}')
for b in range(8):
    kaydet(((xt[b] + 1) / 2).clamp(0, 1), f'uretim_son_{b}')
with torch.no_grad():
    g = ((xt + 1) / 2).clamp(0, 1)
    sonuc['uretim_tahmin'] = [SINIF_TR[i] for i in f(g).argmax(1).tolist()]
json.dump(sonuc, open(f'{C}/d5.json', 'w'))
print('bitti', sonuc['uretim_tahmin'])
