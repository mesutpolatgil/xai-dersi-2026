"""ViT içini açan araçlar: attention ağırlıklarını saklama, head çıktısını değiştirme,
Chefer ve ark. (2021) tarzı alaka (relevancy) haritası."""
import types, torch
import torch.nn.functional as F

def kanca_tak(model):
    """Görüntü kodlayıcısının her bloğuna kayıt ve müdahale yeteneği ekler."""
    bloklar = model.visual.transformer.resblocks
    durum = {'kaydet': False, 'attn': [None] * len(bloklar), 'head_cikti': [None] * len(bloklar),
             'degistir': {}}  # degistir[(katman, head)] = (L, d) ortalama çıktı tensörü veya 0

    def yeni_forward(self, x, k_x=None, v_x=None, attn_mask=None, _i=None):
        N, L, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        h, d = self.num_heads, self.head_dim
        q = q.reshape(N, L, h, d).transpose(1, 2)
        k = k.reshape(N, L, h, d).transpose(1, 2)
        v = v.reshape(N, L, h, d).transpose(1, 2)
        a = ((q * self.scale) @ k.transpose(-1, -2)).softmax(-1)
        if durum['kaydet']:
            if a.requires_grad:
                a.retain_grad()
            durum['attn'][_i] = a
        o = a @ v  # N,h,L,d
        if durum['kaydet']:
            durum['head_cikti'][_i] = o.detach()
        if durum['degistir']:
            o = o.clone()
            for (li, hi), ort in durum['degistir'].items():
                if li == _i:
                    o[:, hi] = ort
        o = o.transpose(1, 2).reshape(N, L, C)
        return self.out_proj(o)

    for i, b in enumerate(bloklar):
        assert b.attn.logit_scale is None and b.attn.head_scale is None
        b.attn.forward = types.MethodType(lambda self, x, k_x=None, v_x=None, attn_mask=None, _i=i:
                                          yeni_forward(self, x, k_x, v_x, attn_mask, _i), b.attn)
    return durum


def alaka_haritasi(model, durum, x, hedef_fn):
    """Chefer ve ark. 2021: R = Π (I + mean_h (∇A ⊙ A)+). CLS satırı → 7x7.
    hedef_fn(emb) -> skaler (örn. bir metinle kosinüs)."""
    from ortak import goruntu_emb
    durum['kaydet'] = True
    x = x.clone().requires_grad_(True)
    s = hedef_fn(goruntu_emb(model, x))
    model.zero_grad()
    s.backward()
    durum['kaydet'] = False
    A0 = durum['attn'][0]
    L = A0.shape[-1]
    R = torch.eye(L).unsqueeze(0).repeat(x.shape[0], 1, 1)
    for a in durum['attn']:
        g = a.grad
        cam = (g * a).clamp(min=0).mean(1)
        R = R + cam @ R
    harita = R[:, 0, 1:]
    n = int(round((L - 1) ** 0.5))
    harita = harita.reshape(-1, 1, n, n)
    harita = harita / (harita.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return harita.detach()


def son_katman_cls_attn(durum):
    """Son katmanda CLS'in yamalara dikkati, head ortalaması → 7x7."""
    a = durum['attn'][-1].detach()
    m = a[:, :, 0, 1:].mean(1)
    n = int(round(m.shape[-1] ** 0.5))
    return m.reshape(-1, 1, n, n)
