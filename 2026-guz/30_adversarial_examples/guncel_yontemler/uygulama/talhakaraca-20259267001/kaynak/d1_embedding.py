"""Deney 1: Embedding nedir? Vektör, harita, benzerlik, arama, sıfır-atış."""
import os, json, torch, numpy as np
import ortak
from open_clip import IMAGENET_CLASSNAMES

C = ortak.CIKTI
os.makedirs(f'{C}/img', exist_ok=True)
os.makedirs(f'{C}/thumb', exist_ok=True)
veri = ortak.imagenette(20)
etiket = np.array([l for _, l in veri])
X = torch.stack([ortak.pil2t(ortak.yukle_goruntu(y)) for y, _ in veri])
for i, (y, _) in enumerate(veri):
    ortak.yukle_goruntu(y, 72).save(f'{C}/thumb/{i}.jpg', quality=80)

sonuc = {'n': len(veri), 'etiket': etiket.tolist(), 'dosya': [os.path.basename(y) for y, _ in veri]}
for ad in ['clip', 'fare', 'tecoa']:
    m = ortak.model_yukle(ad)
    T = ortak.sinif_embleri(m)
    with torch.no_grad():
        E = torch.cat([ortak.goruntu_emb(m, X[i:i + 25]) for i in range(0, len(X), 25)])
    np.save(f'{C}/emb_{ad}.npy', E.numpy())
    np.save(f'{C}/temb_{ad}.npy', T.numpy())
    acc = ((E @ T.t()).argmax(1).numpy() == etiket).mean()
    sonuc[f'acc_{ad}'] = float(acc)
    print(ad, 'sıfır-atış doğruluk', acc)
    if ad == 'clip':
        Ec, Tc, mc = E, T, m

# Demo görüntüleri: her sınıftan doğru sınıflanan ilk görüntü
P = (Ec @ Tc.t()).softmax(-1) if False else (100 * Ec @ Tc.t()).softmax(-1)
demo = []
for c in range(10):
    idx = [i for i in range(len(veri)) if etiket[i] == c and P[i].argmax() == c]
    demo.append(idx[0])
sonuc['demo'] = demo
for k, i in enumerate(demo):
    ortak.yukle_goruntu(veri[i][0]).save(f'{C}/img/demo{k}.png')

# Vektör şeridi, benzerlik matrisi
sonuc['demo_vektor'] = [[round(float(v), 4) for v in Ec[i]] for i in demo]
sonuc['metin_vektor'] = [[round(float(v), 4) for v in Tc[c]] for c in range(10)]
sonuc['benzerlik'] = [[round(float(v), 4) for v in (Ec[i] @ Tc.t())] for i in demo]
sonuc['olasilik'] = [[round(float(v), 4) for v in P[i]] for i in demo]
sonuc['ii_benzerlik'] = [[round(float(Ec[i] @ Ec[j]), 4) for j in demo] for i in demo]
sonuc['tt_benzerlik'] = [[round(float(Tc[i] @ Tc[j]), 4) for j in range(10)] for i in range(10)]
# Ortalama sınıf içi / sınıflar arası benzerlik
S = (Ec @ Ec.t()).numpy()
ayni = etiket[:, None] == etiket[None, :]
np.fill_diagonal(ayni, False)
farkli = etiket[:, None] != etiket[None, :]
sonuc['sinif_ici'] = float(S[ayni].mean()); sonuc['siniflar_arasi'] = float(S[farkli].mean())
sonuc['goruntu_metin_ort'] = float((Ec @ Tc.t()).mean())

# Metinle arama
sorgular = [('a dog running in the snow', 'karda koşan köpek'), ('a man holding a big fish', 'büyük bir balık tutan adam'),
            ('a church at night', 'gece bir kilise'), ('a colorful parachute over the sea', 'denizin üstünde renkli paraşüt'),
            ('a person playing a brass instrument', 'nefesli çalgı çalan kişi'), ('a green truck on the street', 'sokakta yeşil kamyon')]
Q = ortak.metin_emb(mc, [q for q, _ in sorgular])
sonuc['arama'] = []
for (q, tr), e in zip(sorgular, Q):
    s = Ec @ e
    top = s.argsort(descending=True)[:5]
    sonuc['arama'].append({'en': q, 'tr': tr, 'sonuc': top.tolist(), 'skor': [round(float(s[t]), 3) for t in top]})
# Görüntüyle arama
sonuc['komsu'] = []
for i in demo:
    s = Ec @ Ec[i]
    top = [int(t) for t in s.argsort(descending=True)[1:6]]
    sonuc['komsu'].append({'sorgu': i, 'sonuc': top, 'skor': [round(float(s[t]), 3) for t in top]})

# ImageNet sözlüğüyle "ne düşünüyor?"
V = torch.cat([ortak.metin_emb(mc, [ortak.SABLON.format(c) for c in IMAGENET_CLASSNAMES[i:i + 100]]) for i in range(0, 1000, 100)])
torch.save(V, f'{C}/sozluk_clip.pt')
sonuc['sozluk_demo'] = []
for i in demo:
    s = V @ Ec[i]
    top = s.argsort(descending=True)[:5]
    sonuc['sozluk_demo'].append([[IMAGENET_CLASSNAMES[t], round(float(s[t]), 3)] for t in top])

# 2B harita: modaliteleri ayrı merkezle, ortak PCA
from sklearn.decomposition import PCA
mi, mt = Ec.mean(0), Tc.mean(0)
Z = torch.cat([Ec - mi, Tc - mt]).numpy()
pca = PCA(2).fit(Z[:len(Ec)])
Y = pca.transform(Z)
np.save(f'{C}/pca.npy', np.stack([pca.components_[0], pca.components_[1]]))
np.save(f'{C}/merkez.npy', np.stack([mi.numpy(), mt.numpy()]))
from sklearn.manifold import TSNE
Yt = TSNE(2, metric='cosine', init='pca', random_state=0, perplexity=25).fit_transform(Ec.numpy())
sonuc['harita_pca'] = [[round(float(a), 4), round(float(b), 4)] for a, b in Y]
sonuc['harita_tsne'] = [[round(float(a), 3), round(float(b), 3)] for a, b in Yt]
sonuc['pca_var'] = pca.explained_variance_ratio_.tolist()
# Modalite boşluğu: merkezlemeden önce
pca0 = PCA(2).fit(torch.cat([Ec, Tc]).numpy())
Y0 = pca0.transform(torch.cat([Ec, Tc]).numpy())
sonuc['harita_ham'] = [[round(float(a), 4), round(float(b), 4)] for a, b in Y0]
json.dump(sonuc, open(f'{C}/d1.json', 'w'))
print('sınıf içi', sonuc['sinif_ici'], 'arası', sonuc['siniflar_arasi'], 'pca var', sonuc['pca_var'])
