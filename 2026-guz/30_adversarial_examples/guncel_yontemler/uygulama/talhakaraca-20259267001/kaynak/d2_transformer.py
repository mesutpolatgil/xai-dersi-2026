"""Deney 2: ViT içinde ne oluyor? Alaka haritası ve XAI ile seçilen tek yama saldırısı."""
import os, sys, json, torch, numpy as np
import torch.nn.functional as F
import ortak, vit_araclar

C = ortak.CIKTI
N_SINIF = int(sys.argv[1]) if len(sys.argv) > 1 else 10
EPSLER = [float(e) / 255 for e in sys.argv[2].split(',')] if len(sys.argv) > 2 else [16 / 255, 64 / 255]
ADIM = 20
m = ortak.model_yukle('clip'); T = ortak.sinif_embleri(m)
durum = vit_araclar.kanca_tak(m)
veri = ortak.imagenette(N_SINIF, tohum=1)
X = torch.stack([ortak.pil2t(ortak.yukle_goruntu(y)) for y, _ in veri])
Y = torch.tensor([l for _, l in veri])


def harita(x, y):
    return vit_araclar.alaka_haritasi(m, durum, x, lambda e: (e * T[y]).sum())


def yama_maskesi(idx):
    M = torch.zeros(len(idx), 1, 224, 224)
    for b, i in enumerate(idx):
        r, c = divmod(int(i), 7)
        M[b, :, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = 1
    return M


def yama_saldirisi(x, y, M, eps):
    d = torch.zeros_like(x, requires_grad=True)
    alfa = eps / 5
    for _ in range(ADIM):
        k = F.cross_entropy(ortak.logits(m, x + d * M, T), y)
        g, = torch.autograd.grad(k, d)
        with torch.no_grad():
            d += alfa * g.sign()
            d.clamp_(-eps, eps)
            d.copy_((x + d * M).clamp(0, 1) - x)
    return (x + d * M).detach()


sonuc = {'eps': EPSLER, 'n': len(veri)}
H = torch.cat([harita(X[i:i + 10], Y[i:i + 10]) for i in range(0, len(X), 10)])
duz = H.flatten(1)
g = torch.Generator().manual_seed(0)
secim = {'en_yuksek': duz.argmax(1), 'rastgele': torch.randint(0, 49, (len(X),), generator=g), 'en_dusuk': duz.argmin(1)}
for eps in EPSLER:
    for ad, idx in secim.items():
        basari = 0
        for i in range(0, len(X), 10):
            xa = yama_saldirisi(X[i:i + 10], Y[i:i + 10], yama_maskesi(idx[i:i + 10]), eps)
            with torch.no_grad():
                basari += (ortak.logits(m, xa, T).argmax(1) != Y[i:i + 10]).sum().item()
        sonuc[f'{ad}_{round(eps * 255)}'] = basari / len(X)
        print(f'eps={eps * 255:.0f}/255 {ad}: başarı {basari / len(X):.2f}', flush=True)
json.dump(sonuc, open(f'{C}/d2_{N_SINIF}.json', 'w'))
