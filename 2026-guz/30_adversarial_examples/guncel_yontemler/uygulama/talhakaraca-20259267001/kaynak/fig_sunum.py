"""Sunum için gerçek sonuçlardan şekiller (koyu tema)."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
import ortak

C = ortak.CIKTI
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sekil'); os.makedirs(OUT, exist_ok=True)
BG = '#1A1F27'; FG = '#D5D9DF'; MUTED = '#8B94A1'; GRID = '#2A313B'; CORAL = '#FF6B4A'; TEAL = '#5EC4B6'; AMBER = '#F2B84B'
PAL = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
plt.rcParams.update({'font.family': 'Inter', 'font.size': 15, 'axes.facecolor': BG, 'figure.facecolor': BG,
                     'axes.edgecolor': GRID, 'axes.labelcolor': FG, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'text.color': FG, 'axes.spines.top': False, 'axes.spines.right': False})
TR = ortak.TR
d1, d3, d4, d5, d6 = [json.load(open(f'{C}/{k}.json')) for k in ['d1', 'd3', 'd4', 'd5', 'd6']]
d2 = json.load(open(f'{C}/d2_10.json'))
im = lambda p: np.asarray(Image.open(p).convert('RGB'))


def save(f, n):
    f.savefig(f'{OUT}/{n}.png', dpi=160, facecolor=BG); plt.close(f)


def kapat(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def isi_ust(ax, h, alfa=0.75):
    h = np.array(h)
    rgba = np.zeros((7, 7, 4)); rgba[..., 0] = 1; rgba[..., 1] = (220 - 180 * h) / 255; rgba[..., 2] = 40 / 255; rgba[..., 3] = alfa * h
    ax.imshow(np.zeros((224, 224, 3)), alpha=0.35)
    ax.imshow(rgba, extent=(0, 224, 224, 0), interpolation='nearest')


# 1 Vektör şeridi
k = 4
f = plt.figure(figsize=(11, 4.4))
ax = f.add_axes([0.0, 0.08, 0.27, 0.84]); ax.imshow(im(f'{C}/img/demo{k}.png')); kapat(ax); ax.set_title('Görüntü: kilise', fontsize=14, color=MUTED)
v = np.array(d1['demo_vektor'][k]).reshape(8, 64); t = np.array(d1['metin_vektor'][k]).reshape(8, 64); t2 = np.array(d1['metin_vektor'][8]).reshape(8, 64)
for j, (arr, ad) in enumerate([(v, 'E(görüntü) · 512 sayı'), (t, 'E("a photo of a church.")'), (t2, 'E("a photo of a golf ball.")')]):
    a = f.add_axes([0.32, 0.68 - j * 0.31, 0.66, 0.22]); a.imshow(arr, cmap='RdBu_r', vmin=-0.12, vmax=0.12, aspect='auto'); kapat(a)
    a.set_title(ad, fontsize=13, loc='left', color=FG)
save(f, 'vektor')
cos_k = float(np.dot(d1['demo_vektor'][k], d1['metin_vektor'][k])); cos_g = float(np.dot(d1['demo_vektor'][k], d1['metin_vektor'][8]))

# 2 Harita (temiz + metinler)
def harita_ciz(ax, saldirilar=False, sec=None):
    h = d6['harita']; P = np.array(h['temiz']); e = np.array(d1['etiket'])
    for c in range(10):
        ax.scatter(*P[e == c].T, s=26, color=PAL[c], alpha=0.35 if saldirilar else 0.9, lw=0)
    for c, p in enumerate(h['metin']):
        ax.scatter(*p, marker='*', s=420, color=PAL[c], edgecolor='white', lw=1.2, zorder=5)
        if not saldirilar:
            ax.text(p[0] + 0.015, p[1] + 0.012, f'"{TR[c]}"', fontsize=12, color=FG, zorder=6)
    kapat(ax); ax.set_facecolor('#161B22')
f, ax = plt.subplots(figsize=(9, 7)); harita_ciz(ax); ax.invert_yaxis(); f.tight_layout(); save(f, 'harita')

# 3 Benzerlik matrisi
M = np.array(d1['benzerlik'])
f, ax = plt.subplots(figsize=(8.4, 7))
ax.imshow(M, cmap='viridis', vmin=0.1, vmax=0.35)
for r in range(10):
    for c in range(10):
        ax.text(c, r, f'{M[r, c]:.2f}', ha='center', va='center', fontsize=10, color='black' if M[r, c] > 0.27 else 'white')
ax.set_xticks(range(10)); ax.set_xticklabels([f'"{x}"' for x in TR], rotation=55, ha='right', fontsize=11)
ax.set_yticks(range(10)); ax.set_yticklabels([f'görüntü: {x}' for x in TR], fontsize=11)
ax.set_xlabel('metin embedding\'i'); f.tight_layout(); save(f, 'matris')

# 4 Arama
q = d1['arama'][0]; q2 = d1['arama'][3]
f, axs = plt.subplots(2, 5, figsize=(11, 5))
for row, qq in enumerate([q, q2]):
    for j, i in enumerate(qq['sonuc']):
        a = axs[row, j]; a.imshow(im(f'{C}/thumb/{i}.jpg')); kapat(a); a.set_title(f'{qq["skor"][j]:.3f}', fontsize=12, color=MUTED)
    axs[row, 0].text(-10, 36, f'"{qq["en"]}"', ha='right', va='center', fontsize=13, color=AMBER)
f.subplots_adjust(left=0.33, right=0.99, wspace=0.08, hspace=0.3); save(f, 'arama')

# 5 Difüzyon ileri + üretim
f, axs = plt.subplots(2, 6, figsize=(12, 4.6))
for j, t in enumerate(d5['ileri_t']):
    a = axs[0, j]; a.imshow(im(f'{C}/dif/ileri_{t}.png')); kapat(a); a.set_title(f't = {t}', fontsize=13, color=MUTED)
for j, kk in enumerate(['0', '10', '25', '40', 'son']):
    a = axs[1, j]; a.imshow(im(f'{C}/dif/uretim_{kk}_1.png')); kapat(a); a.set_title('adım ' + ('50' if kk == 'son' else kk), fontsize=13, color=MUTED)
a = axs[1, 5]; a.imshow(im(f'{C}/dif/uretim_son_3.png')); kapat(a); a.set_title('başka örnek', fontsize=13, color=MUTED)
f.subplots_adjust(left=0.1, right=0.99, wspace=0.06, hspace=0.3)
f.text(0.02, 0.72, 'İleri\n(gürültü)', fontsize=13, color=AMBER); f.text(0.02, 0.25, 'Ters\n(üretim)', fontsize=13, color=TEAL)
save(f, 'difuzyon')

# 6 DiffPure örnekleri
f, axs = plt.subplots(3, 4, figsize=(8.6, 7))
for r, i in enumerate([0, 3, 8]):
    dd = d5['demo'][i]
    for c, (kk, l, pk) in enumerate([('temiz', 'Temiz', 'p_temiz'), ('fark', 'Perturbation ×', None), ('saldiri', 'Saldırı', 'p_saldiri'), ('saldiri_temiz', 'Temizlenmiş', 'p_temizlenmis')]):
        a = axs[r, c]; a.imshow(im(f'{C}/dif/{kk}{i}.png')); kapat(a)
        if r == 0:
            a.set_title(l, fontsize=14, color=MUTED, pad=24)
        if pk:
            p = dd[pk]; am = int(np.argmax(p)); a.text(64, -6, d5['siniflar'][am], ha='center', fontsize=12, color=TEAL if am == dd['y'] else CORAL)
f.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05, hspace=0.25); save(f, 'diffpure')

# 7 ViT yama + alaka
f, axs = plt.subplots(2, 3, figsize=(9, 6.2))
for j, k in enumerate([1, 4, 9]):
    a = axs[0, j]; a.imshow(im(f'{C}/img/demo{k}.png'))
    for g in range(1, 7):
        a.axhline(g * 32, color='white', lw=0.8, alpha=0.8); a.axvline(g * 32, color='white', lw=0.8, alpha=0.8)
    kapat(a)
    a = axs[1, j]; a.imshow(im(f'{C}/img/demo{k}.png')); isi_ust(a, d6['alaka']['temiz'][k]); kapat(a); a.set_xlabel(f'"{TR[k]}" için relevancy', fontsize=12, color=MUTED)
    a.xaxis.label.set_visible(True)
f.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.06, wspace=0.05, hspace=0.12); save(f, 'vit')

# 8 Tipografik örnekler (temiz | yazılı | alaka)
ornek = sorted(range(10), key=lambda i: -d3['demo'][i]['p_yazili'][d3['demo'][i]['hedef']])[:3]
f, axs = plt.subplots(3, 3, figsize=(8.2, 8.4))
for r, i in enumerate(ornek):
    dd = d3['demo'][i]
    for c, (src, l) in enumerate([(f'tipotemiz{i}', 'Temiz'), (f'tipo{i}', 'Yazılı'), (f'tipo{i}', 'Relevancy map')]):
        a = axs[r, c]; a.imshow(im(f'{C}/img/{src}.png')); kapat(a)
        if c == 2:
            isi_ust(a, dd['harita_yazili'])
        if r == 0:
            a.set_title(l, fontsize=14, color=MUTED, pad=26)
        p = dd['p_temiz'] if c == 0 else dd['p_yazili']; am = int(np.argmax(p))
        if c < 2:
            a.text(112, -6, f'{TR[am]} · %{p[am] * 100:.0f}', ha='center', fontsize=12, color=TEAL if am == dd['y'] else CORAL)
f.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.05, hspace=0.2); save(f, 'tipo')
json.dump({'tipo_ornek': ornek, 'cos_kilise': cos_k, 'cos_golf': cos_g}, open(f'{OUT}/bilgi.json', 'w'))

# 9 Muz saldırısı
f, axs = plt.subplots(2, 3, figsize=(9, 6.4))
for r, k in enumerate([4, 6]):
    hh = d4['hedefli']['clip']
    for c, (src, l) in enumerate([(f'demo{k}', 'Orijinal'), (f'muzfark_clip_8_{k}', 'Perturbation (±8/255, büyütülmüş)'), (f'muz_clip_8_{k}', 'Saldırı sonrası')]):
        a = axs[r, c]; a.imshow(im(f'{C}/img/{src}.png')); kapat(a)
        if r == 0:
            a.set_title(l, fontsize=13, color=MUTED, pad=10)
    w0 = hh['0'][k]['sozluk'][0][0]; w1 = hh['8'][k]['sozluk'][0][0]
    axs[r, 0].set_xlabel(f'en yakın metin: {w0}', fontsize=12, color=TEAL); axs[r, 2].set_xlabel(f'en yakın metin: {w1}', fontsize=12, color=CORAL)
    axs[r, 0].xaxis.label.set_visible(True); axs[r, 2].xaxis.label.set_visible(True)
f.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06, wspace=0.05, hspace=0.25); save(f, 'muz')

# 10 cos(muz) – ε eğrisi, CLIP vs FARE
f, ax = plt.subplots(figsize=(7.5, 5))
E = [0, 1, 2, 4, 8]
for ad, renk, l in [('clip', CORAL, 'CLIP'), ('fare', TEAL, 'FARE (gürbüz)')]:
    y = [np.mean([s['cos_hedef'] for s in d4['hedefli'][ad][str(e)]]) for e in E]
    ax.plot(range(5), y, '-o', color=renk, lw=3, label=l)
    oran = [np.mean(['banana' in s['sozluk'][0][0].lower() for s in d4['hedefli'][ad][str(e)]]) for e in E]
    for j in range(1, 5):
        ax.text(j, y[j] + 0.012, f'{oran[j] * 100:.0f}%', ha='center', fontsize=11, color=renk)
ax.set_xticks(range(5)); ax.set_xticklabels([f'{e}/255' if e else '0' for e in E]); ax.set_xlabel('ε'); ax.set_ylabel('cos(görüntü, "muz")')
ax.grid(axis='y', color=GRID); ax.legend(frameon=False); ax.set_title('Yüzde: en yakın metni "banana" olan görüntü oranı', fontsize=12, color=MUTED)
f.tight_layout(); save(f, 'muz_eps')

# 11 Haritada hareket
f, ax = plt.subplots(figsize=(9, 7)); harita_ciz(ax, saldirilar=True)
h = d6['harita']; dm = d1['demo']
for kk in range(10):
    a = h['temiz'][dm[kk]]; b = h['muz'][kk]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=16, color=CORAL, lw=1.8, zorder=7))
    ax.scatter(*a, s=60, color=PAL[kk], edgecolor='white', zorder=8)
m = h['muz_metin'][0]; ax.scatter(*m, marker='*', s=900, color=AMBER, edgecolor='white', lw=1.5, zorder=9)
ax.text(m[0] + 0.02, m[1], '"muz"', fontsize=15, color=AMBER, zorder=9, fontweight='bold')
ax.invert_yaxis(); f.tight_layout(); save(f, 'harita_muz')

# 12 Alaka: temiz vs saldırı
f, axs = plt.subplots(2, 3, figsize=(9, 6.2))
for j, kk in enumerate([0, 4, 6]):
    a = axs[0, j]; a.imshow(im(f'{C}/img/demo{kk}.png')); isi_ust(a, d6['alaka']['temiz'][kk]); kapat(a)
    a = axs[1, j]; a.imshow(im(f'{C}/img/muz_clip_8_{kk}.png')); isi_ust(a, d6['alaka']['muz'][kk]); kapat(a)
axs[0, 0].set_ylabel('Temiz\n(doğru sınıf)', fontsize=13); axs[1, 0].set_ylabel('Saldırı\n("muz")', fontsize=13)
for a in axs[:, 0]:
    a.yaxis.label.set_visible(True)
f.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.02, wspace=0.05, hspace=0.06); save(f, 'alaka_muz')

# 13 Görüntüden görüntüye
f, axs = plt.subplots(2, 3, figsize=(9, 6.4))
for r, kk in enumerate([0, 6]):
    g = d4['g2g'][kk]
    for c, (src, l) in enumerate([(f'demo{kk}', 'Kaynak'), (f'g2g_{kk}', 'Kaynak + δ'), (f'demo{g["hedef"]}', 'Hedef görüntü')]):
        a = axs[r, c]; a.imshow(im(f'{C}/img/{src}.png')); kapat(a)
        if r == 0:
            a.set_title(l, fontsize=13, color=MUTED)
    am = int(np.argmax(g['p']))
    axs[r, 1].set_xlabel(f'cos(hedef): {g["cos_once"]:.2f} → {g["cos_sonra"]:.2f} · sınıf: {TR[am]}', fontsize=12, color=CORAL); axs[r, 1].xaxis.label.set_visible(True)
f.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.07, wspace=0.05, hspace=0.25); save(f, 'g2g')

# 14 Head etkileri
E20 = d3['head_etkileri'][:20]; t0 = d3['arama_taban']['yazi_kazandi']
f, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(20), [(t0 - e['yazi_kazandi']) * 100 for e in E20], color=[CORAL if j < 12 else MUTED for j in range(20)])
ax.set_xticks(range(20)); ax.set_xticklabels([f"L{e['katman']}.H{e['head']}" for e in E20], rotation=60, fontsize=11)
ax.set_ylabel('"yazı kazandı" düşüşü (puan)'); ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
f.tight_layout(); save(f, 'headler')

# 15 Ablation eğrisi
eg = d3['ablation_egrisi']
f, ax = plt.subplots(figsize=(8, 5))
ax.plot([e['k'] for e in eg], [e['yazi_kazandi'] * 100 for e in eg], '-o', color=CORAL, lw=3, label='yazı kazandı')
ax.plot([e['k'] for e in eg], [e['temiz_dogruluk'] * 100 for e in eg], '-o', color=TEAL, lw=3, label='temiz doğruluk')
ax.plot([e['k'] for e in eg], [e['yazili_dogruluk'] * 100 for e in eg], '--', color=AMBER, lw=2, label='yazılı görüntüde doğruluk')
ax.set_xlabel('kapatılan head sayısı'); ax.set_ylabel('%'); ax.set_ylim(0, 105); ax.grid(axis='y', color=GRID); ax.legend(frameon=False, loc='center right')
f.tight_layout(); save(f, 'onarim')

# 16 Head dikkati
b = d3['head_etkileri'][0]
f, axs = plt.subplots(1, 3, figsize=(9, 3.4))
for j, i in enumerate(ornek):
    a = axs[j]; a.imshow(im(f'{C}/img/tipo{i}.png')); isi_ust(a, d3['demo'][i]['head_dikkat'], 0.85); kapat(a)
f.suptitle(f"L{b['katman']}.H{b['head']} head'inin attention'ı", fontsize=13, color=MUTED); f.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.05); save(f, 'head_dikkat')

# 17 Savunma eğrisi
f, ax = plt.subplots(figsize=(8, 5.2)); EE = ['0'] + [str(e) for e in d4['eps']]
for ad, renk, l in [('clip', CORAL, 'CLIP'), ('fare', TEAL, 'FARE'), ('tecoa', AMBER, 'TeCoA')]:
    ax.plot(range(len(EE)), [d4[f'saglam_{ad}'][e] * 100 for e in EE], '-o', color=renk, lw=3, label=l)
ax.plot(range(len(EE)), [d4['transfer_b16'][e] * 100 for e in EE], ':', color=MUTED, lw=2, label='Transfer: CLIP B/32 → B/16')
ax.set_xticks(range(len(EE))); ax.set_xticklabels([f'{e}/255' if e != '0' else '0' for e in EE]); ax.set_xlabel('ε (L∞)'); ax.set_ylabel('doğruluk %'); ax.set_ylim(0, 105)
ax.grid(axis='y', color=GRID); ax.legend(frameon=False, fontsize=12); f.tight_layout(); save(f, 'savunma')

# 18 Tipografik: modellere göre
f, ax = plt.subplots(figsize=(7.5, 4.6))
ad = ['CLIP', 'FARE', 'TeCoA', 'CLIP +\n12 head kapalı']; v = [d3['clip']['yazi_kazandi'], d3['fare']['yazi_kazandi'], d3['tecoa']['yazi_kazandi'], eg[12]['yazi_kazandi']]
ax.bar(range(4), [x * 100 for x in v], color=[CORAL, CORAL, CORAL, TEAL])
for j, x in enumerate(v):
    ax.text(j, x * 100 + 1.5, f'%{x * 100:.0f}', ha='center', fontsize=14)
ax.set_xticks(range(4)); ax.set_xticklabels(ad, fontsize=13); ax.set_ylabel('yazı kazandı %'); ax.set_ylim(0, 75); ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
f.tight_layout(); save(f, 'tipo_model')
print('şekiller hazır', os.listdir(OUT))
