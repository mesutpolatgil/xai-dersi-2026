"""Deney 6: Ortak embedding haritası (temiz + saldırılar + metinler) ve saldırıların alaka haritaları."""
import json, torch, numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import ortak, vit_araclar

C = ortak.CIKTI
d1, d4 = json.load(open(f'{C}/d1.json')), json.load(open(f'{C}/d4.json'))
m = ortak.model_yukle('clip'); T = ortak.sinif_embleri(m)
durum = vit_araclar.kanca_tak(m)
E = torch.from_numpy(np.load(f'{C}/emb_clip.npy'))
yukle = lambda p: ortak.pil2t(Image.open(p).convert('RGB'))
XD = torch.stack([yukle(f'{C}/img/demo{k}.png') for k in range(10)])
XM = torch.stack([yukle(f'{C}/img/muz_clip_8_{k}.png') for k in range(10)])
XT = torch.stack([yukle(f'{C}/img/tipo{k}.png') for k in range(10)])
XG = torch.stack([yukle(f'{C}/img/g2g_{k}.png') for k in range(10)])
with torch.no_grad():
    EM, ET, EG = ortak.goruntu_emb(m, XM), ortak.goruntu_emb(m, XT), ortak.goruntu_emb(m, XG)
muz = torch.tensor(d4['hedef_metin_emb'])
mi, mt = E.mean(0), T.mean(0)
Z = torch.cat([E - mi, EM - mi, ET - mi, EG - mi, T - mt, (muz - mt).unsqueeze(0)])
Z = torch.nn.functional.normalize(Z, dim=-1).numpy()
Y = TSNE(2, metric='cosine', init='pca', random_state=0, perplexity=25).fit_transform(Z)
Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))
n = len(E)
parca = {'temiz': (0, n), 'muz': (n, n + 10), 'tipo': (n + 10, n + 20), 'g2g': (n + 20, n + 30),
         'metin': (n + 30, n + 40), 'muz_metin': (n + 40, n + 41)}
harita = {k: [[round(float(a), 4), round(float(b), 4)] for a, b in Y[s:e]] for k, (s, e) in parca.items()}

# Alaka haritaları: temiz (doğru sınıf), muz saldırısı (muz metni), görüntüden görüntüye
def alaka(x, hedef):
    return vit_araclar.alaka_haritasi(m, durum, x, lambda e: (e * hedef).sum(-1).sum())[:, 0]
muz_t = ortak.metin_emb(m, [d4['hedef_metin'][0]])[0]
H = {'temiz': alaka(XD, T), 'muz': alaka(XM, muz_t.expand(10, -1))}
H = {k: [[[round(float(v), 3) for v in r] for r in h] for h in hs] for k, hs in H.items()}
# Kosinüs benzerlikleri: sınıf metinlerine ve muz metnine (temiz/saldırı)
with torch.no_grad():
    cos = {'temiz_sinif': (ortak.goruntu_emb(m, XD) @ T.t()).tolist(), 'muz_sinif': (EM @ T.t()).tolist(),
           'temiz_muz': (ortak.goruntu_emb(m, XD) @ muz_t).tolist(), 'muz_muz': (EM @ muz_t).tolist(),
           'tipo_sinif': (ET @ T.t()).tolist()}
json.dump({'harita': harita, 'alaka': H, 'cos': cos,
           'tipo_emb': [[round(float(v), 4) for v in e] for e in ET]}, open(f'{C}/d6.json', 'w'))
print('bitti')
