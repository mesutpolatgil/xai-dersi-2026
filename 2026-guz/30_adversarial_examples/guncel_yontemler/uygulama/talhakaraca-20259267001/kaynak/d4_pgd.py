"""Deney 4: Embedding'e PGD saldırısı. Hedefsiz istatistik, hedefli metin/görüntü saldırısı,
modele göre karşılaştırma (CLIP, FARE, TeCoA) ve başka bir modele transfer."""
import os, json, torch, numpy as np
import torch.nn.functional as F
import open_clip
import ortak
from open_clip import IMAGENET_CLASSNAMES

C = ortak.CIKTI
EPS = [1, 2, 4, 8]
sonuc = {'eps': EPS}
veri = ortak.imagenette(10, tohum=5)
X = torch.stack([ortak.pil2t(ortak.yukle_goruntu(y)) for y, _ in veri])
Y = torch.tensor([l for _, l in veri])
d1 = json.load(open(f'{C}/d1.json'))
demo_yol = [f'{C}/img/demo{k}.png' for k in range(10)]
XD = torch.stack([ortak.pil2t(__import__('PIL.Image', fromlist=['Image']).open(p).convert('RGB')) for p in demo_yol])
YD = torch.arange(10)
HEDEF_METIN = ('a photo of a banana.', 'muz')

# Transfer için ikinci model: OpenAI ViT-B/16
m2, _, _ = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='openai'); m2.eval()
for p in m2.parameters():
    p.requires_grad_(False)
T2 = ortak.sinif_embleri(m2)
adv_clip = {}

for ad in ['clip', 'fare', 'tecoa']:
    m = ortak.model_yukle(ad); T = ortak.sinif_embleri(m)
    r = {}
    with torch.no_grad():
        r['0'] = (ortak.logits(m, X, T).argmax(1) == Y).float().mean().item()
    for e in EPS:
        dogru, transfer = 0, 0
        for i in range(0, len(X), 20):
            x, y = X[i:i + 20], Y[i:i + 20]
            xa = ortak.pgd(lambda z: F.cross_entropy(ortak.logits(m, z, T), y), x, e / 255, 10)
            with torch.no_grad():
                dogru += (ortak.logits(m, xa, T).argmax(1) == y).sum().item()
                if ad == 'clip':
                    transfer += (ortak.logits(m2, xa, T2).argmax(1) == y).sum().item()
        r[str(e)] = dogru / len(X)
        if ad == 'clip':
            sonuc.setdefault('transfer_b16', {})[str(e)] = transfer / len(X)
        print(ad, e, r[str(e)], flush=True)
    sonuc[f'saglam_{ad}'] = r
with torch.no_grad():
    sonuc['transfer_b16']['0'] = (ortak.logits(m2, X, T2).argmax(1) == Y).float().mean().item()
print('transfer', sonuc['transfer_b16'], flush=True)

# Hedefli metin saldırısı: demo görüntülerinin embedding'ini "muz" metnine taşı
V = torch.load(f'{C}/sozluk_clip.pt')
sonuc['hedefli'] = {}
for ad in ['clip', 'fare']:
    m = ortak.model_yukle(ad); T = ortak.sinif_embleri(m)
    t = ortak.metin_emb(m, [HEDEF_METIN[0]])[0]
    Vm = V if ad == 'clip' else torch.cat([ortak.metin_emb(m, [ortak.SABLON.format(c) for c in IMAGENET_CLASSNAMES[i:i + 100]]) for i in range(0, 1000, 100)])
    kayit = {}
    for e in [0] + EPS:
        if e == 0:
            xa = XD.clone()
        else:
            xa = ortak.pgd(lambda z: (ortak.goruntu_emb(m, z) @ t).sum(), XD, e / 255, 50, alfa=e / 255 / 10)
        with torch.no_grad():
            E = ortak.goruntu_emb(m, xa)
            P = (100 * E @ T.t()).softmax(-1)
        satir = []
        for k in range(10):
            s = Vm @ E[k]
            top = s.argsort(descending=True)[:5]
            satir.append({'cos_hedef': round(float(E[k] @ t), 4), 'p': [round(float(v), 4) for v in P[k]],
                          'sozluk': [[IMAGENET_CLASSNAMES[j], round(float(s[j]), 3)] for j in top],
                          'emb': [round(float(v), 4) for v in E[k]]})
            if ad == 'clip' or e == 8:
                ortak.t2pil(xa[k]).save(f'{C}/img/muz_{ad}_{e}_{k}.png')
                if e > 0:
                    dlt = (xa[k] - XD[k]) / (2 * e / 255) + 0.5
                    ortak.t2pil(dlt).save(f'{C}/img/muzfark_{ad}_{e}_{k}.png')
        kayit[str(e)] = satir
        print(ad, 'hedefli', e, np.mean([s['cos_hedef'] for s in satir]), flush=True)
    sonuc['hedefli'][ad] = kayit
sonuc['hedef_metin'] = HEDEF_METIN
sonuc['hedef_metin_emb'] = [round(float(v), 4) for v in ortak.metin_emb(ortak.model_yukle('clip'), [HEDEF_METIN[0]])[0]]

# Görüntüden görüntüye: demo k'nın embedding'ini demo (k+3)%10'a taşı (ε=8/255)
m = ortak.model_yukle('clip'); T = ortak.sinif_embleri(m)
with torch.no_grad():
    ED = ortak.goruntu_emb(m, XD)
hedef = ED[[(k + 3) % 10 for k in range(10)]]
xa = ortak.pgd(lambda z: (ortak.goruntu_emb(m, z) * hedef).sum(), XD, 8 / 255, 50, alfa=8 / 255 / 10)
with torch.no_grad():
    E = ortak.goruntu_emb(m, xa); P = (100 * E @ T.t()).softmax(-1)
sonuc['g2g'] = []
for k in range(10):
    ortak.t2pil(xa[k]).save(f'{C}/img/g2g_{k}.png')
    sonuc['g2g'].append({'hedef': (k + 3) % 10, 'cos_once': round(float(ED[k] @ hedef[k]), 4),
                         'cos_sonra': round(float(E[k] @ hedef[k]), 4), 'p': [round(float(v), 4) for v in P[k]],
                         'emb': [round(float(v), 4) for v in E[k]]})
json.dump(sonuc, open(f'{C}/d4.json', 'w'))
print('bitti')
