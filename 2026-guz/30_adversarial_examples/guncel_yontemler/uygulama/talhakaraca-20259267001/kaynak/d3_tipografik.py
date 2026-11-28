"""Deney 3: Tipografik saldırı, XAI ile açığın kaynağını bulma, head ablation ile onarma."""
import os, json, random, torch, numpy as np
from PIL import ImageDraw, ImageFont
import ortak, vit_araclar

C = ortak.CIKTI
os.makedirs(f'{C}/img', exist_ok=True)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
YAZI = ['tench', 'spaniel', 'cassette player', 'chainsaw', 'church', 'french horn', 'garbage truck', 'gas pump',
        'golf ball', 'parachute']


def yaz(im, metin, bos=False):
    im = im.copy()
    dr = ImageDraw.Draw(im)
    l, t, r, b = dr.textbbox((0, 0), metin, font=FONT)
    w, h = r - l, b - t
    x0, y0 = (224 - w) // 2, 150
    dr.rectangle([x0 - 8, y0 - 6, x0 + w + 8, y0 + h + 10], fill=(255, 255, 255))
    if not bos:
        dr.text((x0 - l, y0 - t), metin, fill=(0, 0, 0), font=FONT)
    return im


def set_hazirla(n, tohum):
    rng = random.Random(tohum)
    veri = ortak.imagenette(n, tohum=tohum)
    temiz, yazili, bos, y, hedef = [], [], [], [], []
    for yol, l in veri:
        im = ortak.yukle_goruntu(yol)
        h = rng.choice([c for c in range(10) if c != l])
        temiz.append(ortak.pil2t(im)); yazili.append(ortak.pil2t(yaz(im, YAZI[h])))
        bos.append(ortak.pil2t(yaz(im, YAZI[h], bos=True)))
        y.append(l); hedef.append(h)
    return (torch.stack(temiz), torch.stack(yazili), torch.stack(bos), torch.tensor(y), torch.tensor(hedef))


def tahmin(m, T, X):
    with torch.no_grad():
        return torch.cat([ortak.logits(m, X[i:i + 25], T).argmax(1) for i in range(0, len(X), 25)])


def olc(m, T, s):
    Xt, Xy, Xb, Y, H = s
    pt, py, pb = tahmin(m, T, Xt), tahmin(m, T, Xy), tahmin(m, T, Xb)
    return {'temiz_dogruluk': (pt == Y).float().mean().item(),
            'yazili_dogruluk': (py == Y).float().mean().item(),
            'yazi_kazandi': (py == H).float().mean().item(),
            'bos_etiket_dogruluk': (pb == Y).float().mean().item()}


test = set_hazirla(10, 2)     # 100 görüntü: değerlendirme
arama = set_hazirla(5, 4)     # 50 görüntü: head arama
ort_set = set_hazirla(10, 3)  # ortalama head çıktısı için temiz görüntüler

sonuc = {}
for ad in ['clip', 'fare', 'tecoa']:
    m = ortak.model_yukle(ad); T = ortak.sinif_embleri(m)
    sonuc[ad] = olc(m, T, test)
    print(ad, sonuc[ad], flush=True)

# --- XAI: hangi head'ler yazıyı taşıyor? ---
m = ortak.model_yukle('clip'); T = ortak.sinif_embleri(m)
durum = vit_araclar.kanca_tak(m)
durum['kaydet'] = True
toplam = None
with torch.no_grad():
    for i in range(0, 100, 25):
        ortak.goruntu_emb(m, ort_set[0][i:i + 25])
        o = [h.sum(0) for h in durum['head_cikti']]
        toplam = o if toplam is None else [a + b for a, b in zip(toplam, o)]
durum['kaydet'] = False
ORT = [t / 100 for t in toplam]  # katman başına (h, L, d)

KATMANLAR = list(range(6, 12))
taban = olc(m, T, arama)
etkiler = []
for li in KATMANLAR:
    for hi in range(12):
        durum['degistir'] = {(li, hi): ORT[li][hi]}
        r = olc(m, T, arama)
        etkiler.append({'katman': li, 'head': hi, 'yazi_kazandi': r['yazi_kazandi'], 'temiz': r['temiz_dogruluk']})
durum['degistir'] = {}
etkiler.sort(key=lambda e: e['yazi_kazandi'])
sonuc['arama_taban'] = taban
sonuc['head_etkileri'] = etkiler
print('en etkili headler', etkiler[:8], flush=True)

# Açgözlü birikimli ablation, test setinde
egri = [{'k': 0, **olc(m, T, test)}]
secili = {}
for k, e in enumerate(etkiler[:12], 1):
    secili[(e['katman'], e['head'])] = ORT[e['katman']][e['head']]
    durum['degistir'] = dict(secili)
    egri.append({'k': k, 'head': f"L{e['katman']}.H{e['head']}", **olc(m, T, test)})
    print(egri[-1], flush=True)
durum['degistir'] = {}
sonuc['ablation_egrisi'] = egri

# Demo: görseller, alaka haritası, en etkili head'in dikkati
demo = [k * 10 for k in range(10)]
Xt, Xy, Xb, Y, H = test
sonuc['demo'] = []
en_iyi = etkiler[0]
for j, i in enumerate(demo):
    ortak.t2pil(Xy[i]).save(f'{C}/img/tipo{j}.png')
    ortak.t2pil(Xb[i]).save(f'{C}/img/tipobos{j}.png')
    ortak.t2pil(Xt[i]).save(f'{C}/img/tipotemiz{j}.png')
    with torch.no_grad():
        pt = ortak.logits(m, Xt[i:i + 1], T).softmax(-1)[0]
        py = ortak.logits(m, Xy[i:i + 1], T).softmax(-1)[0]
    pred = int(py.argmax())
    hm = vit_araclar.alaka_haritasi(m, durum, Xy[i:i + 1], lambda e: (e * T[pred]).sum())[0, 0]
    hm_t = vit_araclar.alaka_haritasi(m, durum, Xt[i:i + 1], lambda e: (e * T[int(pt.argmax())]).sum())[0, 0]
    durum['kaydet'] = True
    with torch.no_grad():
        ortak.goruntu_emb(m, Xy[i:i + 1])
    durum['kaydet'] = False
    a = durum['attn'][en_iyi['katman']][0, en_iyi['head'], 0, 1:].reshape(7, 7)
    a = a / a.max()
    durum['degistir'] = dict(secili)
    with torch.no_grad():
        pa = ortak.logits(m, Xy[i:i + 1], T).softmax(-1)[0]
    durum['degistir'] = {}
    sonuc['demo'].append({'y': int(Y[i]), 'hedef': int(H[i]), 'p_temiz': [round(float(v), 4) for v in pt],
                          'p_yazili': [round(float(v), 4) for v in py], 'p_onarilmis': [round(float(v), 4) for v in pa],
                          'harita_yazili': [[round(float(v), 3) for v in r] for r in hm],
                          'harita_temiz': [[round(float(v), 3) for v in r] for r in hm_t],
                          'head_dikkat': [[round(float(v), 3) for v in r] for r in a]})
json.dump(sonuc, open(f'{C}/d3.json', 'w'))
