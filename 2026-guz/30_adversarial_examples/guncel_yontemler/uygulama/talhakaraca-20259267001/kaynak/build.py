"""Sonuçları ve görselleri tek bir HTML dosyasına gömer."""
import os, json, base64, io, glob
from PIL import Image
import ortak

C = ortak.CIKTI
K = os.path.dirname(os.path.abspath(__file__))
from d3_tipografik_sabit import YAZI  # noqa


def b64(yol, fmt='JPEG', q=86, boyut=None):
    im = Image.open(yol).convert('RGB')
    if boyut:
        im = im.resize(boyut, Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, fmt, quality=q) if fmt == 'JPEG' else im.save(buf, fmt, optimize=True)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buf.getvalue()).decode()


IMG = {}
for y in glob.glob(f'{C}/img/*.png'):
    IMG[os.path.basename(y)[:-4]] = b64(y)
for y in glob.glob(f'{C}/thumb/*.jpg'):
    IMG['th' + os.path.basename(y)[:-4]] = b64(y, q=80)
for y in glob.glob(f'{C}/dif/*.png'):
    IMG['dif_' + os.path.basename(y)[:-4]] = b64(y, 'PNG')
gri = Image.new('RGB', (224, 224), (128, 128, 128)); gri.save(f'{C}/gri.png')
IMG['gri'] = b64(f'{C}/gri.png')

D = {'siniflar': ortak.TR, 'en': ortak.EN, 'yazi': YAZI,
     'cifar_siniflar': json.load(open(f'{C}/d5.json'))['siniflar']}
for k in ['d1', 'd3', 'd4', 'd5', 'd6']:
    D[k] = json.load(open(f'{C}/{k}.json'))
D['d2'] = json.load(open(f'{C}/d2_10.json'))
# Gereksiz büyük alanları çıkar
for ad in D['d4']['hedefli']:
    for e in D['d4']['hedefli'][ad]:
        for s in D['d4']['hedefli'][ad][e]:
            s.pop('emb', None)
for g in D['d4']['g2g']:
    g.pop('emb', None)
D['d6'].pop('tipo_emb', None)
for k in ['harita_pca', 'harita_tsne', 'harita_ham']:
    D['d1'].pop(k, None)

s = open(f'{K}/sablon.html').read()
s = s.replace('__DATA__', json.dumps(D, ensure_ascii=False, separators=(',', ':')))
s = s.replace('__IMG__', json.dumps(IMG, separators=(',', ':')))
out = f'{K}/embedding-lab.html'
open(out, 'w').write(s)
print(out, round(len(s) / 1e6, 2), 'MB', len(IMG), 'görsel')
