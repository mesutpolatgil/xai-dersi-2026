"""Deney 3 eki: demo görüntülerinde boş kutu (yazısız) olasılıkları ve embedding."""
import json, torch
from PIL import Image
import ortak

C = ortak.CIKTI
d3 = json.load(open(f'{C}/d3.json'))
m = ortak.model_yukle('clip'); T = ortak.sinif_embleri(m)
X = torch.stack([ortak.pil2t(Image.open(f'{C}/img/tipobos{k}.png').convert('RGB')) for k in range(10)])
with torch.no_grad():
    P = ortak.logits(m, X, T).softmax(-1)
for k in range(10):
    d3['demo'][k]['p_bos'] = [round(float(v), 4) for v in P[k]]
json.dump(d3, open(f'{C}/d3.json', 'w'))
print('bitti')
