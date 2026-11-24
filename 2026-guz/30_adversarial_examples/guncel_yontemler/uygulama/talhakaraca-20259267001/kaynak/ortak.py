"""Ortak yardımcılar: modeller, veri, sıfır-atış sınıflandırıcı, PGD."""
import os, glob, random, torch, numpy as np
import torch.nn.functional as F
import open_clip
from PIL import Image

KOK = os.path.dirname(os.path.abspath(__file__))
VERI = os.path.join(KOK, 'veri')
CIKTI = os.path.join(KOK, 'cikti')
os.makedirs(CIKTI, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() or 1))

# Imagenette sınıfları (klasör adı, İngilizce istem adı, Türkçe ad)
SINIFLAR = [
    ('n01440764', 'tench', 'kadife balığı'),
    ('n02102040', 'English springer spaniel', 'köpek'),
    ('n02979186', 'cassette player', 'kaset çalar'),
    ('n03000684', 'chainsaw', 'motorlu testere'),
    ('n03028079', 'church', 'kilise'),
    ('n03394916', 'French horn', 'korno'),
    ('n03417042', 'garbage truck', 'çöp kamyonu'),
    ('n03425413', 'gas pump', 'akaryakıt pompası'),
    ('n03445777', 'golf ball', 'golf topu'),
    ('n03888257', 'parachute', 'paraşüt'),
]
EN = [s[1] for s in SINIFLAR]
TR = [s[2] for s in SINIFLAR]
SABLON = 'a photo of a {}.'

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

MODELLER = {
    'clip': ('ViT-B-32', 'laion2b_s34b_b79k'),
    'fare': ('hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K', None),
    'tecoa': ('hf-hub:chs20/TeCoA4-ViT-B-32-laion2B-s34B-b79K', None),
}


def model_yukle(ad='clip'):
    arch, pre = MODELLER[ad]
    if pre:
        m, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre)
    else:
        m, _, _ = open_clip.create_model_and_transforms(arch)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


_tok = None
def metin_emb(model, metinler):
    global _tok
    if _tok is None:
        _tok = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        e = model.encode_text(_tok(metinler))
    return F.normalize(e, dim=-1)


def goruntu_emb(model, x):
    """x: [0,1] aralığında 224x224 görüntü tensörü (B,3,224,224)."""
    return F.normalize(model.encode_image((x - MEAN) / STD), dim=-1)


def sinif_embleri(model):
    return metin_emb(model, [SABLON.format(c) for c in EN])


def logits(model, x, T):
    return 100.0 * goruntu_emb(model, x) @ T.t()


def yukle_goruntu(yol, boyut=224):
    im = Image.open(yol).convert('RGB')
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((boyut, boyut), Image.BICUBIC)
    return im


def pil2t(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)


def t2pil(t):
    a = (t.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def imagenette(n_sinif=20, bolum='val', tohum=0):
    """Her sınıftan n_sinif görüntü (yol, etiket)."""
    rng = random.Random(tohum)
    kok = glob.glob(os.path.join(VERI, 'imagenette2-*'))[0]
    out = []
    for i, (kl, _, _) in enumerate(SINIFLAR):
        yollar = sorted(glob.glob(os.path.join(kok, bolum, kl, '*.JPEG')))
        rng.shuffle(yollar)
        out += [(y, i) for y in yollar[:n_sinif]]
    return out


def pgd(f_kayip, x, eps, adim, alfa=None, rastgele=True):
    """f_kayip(x_adv) -> büyütülecek skaler. L∞ topu, [0,1] kırpma."""
    alfa = alfa or max(eps / 4, 1 / 255 / 4)
    d = (torch.rand_like(x) * 2 - 1) * eps if rastgele else torch.zeros_like(x)
    d = ((x + d).clamp(0, 1) - x).requires_grad_(True)
    for _ in range(adim):
        k = f_kayip(x + d)
        g, = torch.autograd.grad(k, d)
        with torch.no_grad():
            d += alfa * g.sign()
            d.clamp_(-eps, eps)
            d.copy_((x + d).clamp(0, 1) - x)
    return (x + d).detach()
