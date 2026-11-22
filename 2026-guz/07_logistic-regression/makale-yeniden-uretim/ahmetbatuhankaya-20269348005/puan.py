"""FasterRisk (Liu ve ark., 2022) yeniden üretimi için yardımcı fonksiyonlar.

Karşılaştırılan yöntemler (hepsi en fazla k öznitelik kullanır):
  - FasterRisk: seyrek ışın araması + çeşitli havuz + yıldız ışını araması ve sıralı yuvarlama
  - Lasso + EKK: L1 lojistik regresyonla k öznitelik seçilir, sonra cezasız lojistik regresyon (sürekli, yuvarlanmamış)
  - Lasso + yuvarlama: aynı katsayılar doğrudan en yakın tam sayıya yuvarlanır
  - Lasso + ölçekle-yuvarla: katsayılar en büyüğü 5 olacak şekilde ölçeklenip yuvarlanır (çarpanlı puan)
Ayrıca üst referans olarak tüm öznitelikleri kullanan cezasız lojistik regresyon.
"""
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer

warnings.filterwarnings("ignore")
SINIR = 5  # puanlar [-5, 5] aralığında (makaledeki gibi)


def _lr(X, y, C=1e6):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def lasso_destek(X, y, k):
    """L1 yolunu tarayıp en fazla k öznitelik seçen en büyük C'nin desteğini döndürür."""
    en_iyi = np.array([], dtype=int)
    for C in np.logspace(-4, 1, 60):
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=5000).fit(X, y)
        destek = np.flatnonzero(np.abs(m.coef_[0]) > 1e-8)
        if len(destek) > k:
            break
        en_iyi = destek
    return en_iyi


def _kesisim_ayarla(skor, y):
    """Tam sayı puanlar sabitken en iyi kesişimi (ölçekli logit için) bulur."""
    m = LogisticRegression(C=1e6, max_iter=5000).fit(skor.reshape(-1, 1), y)
    return m


def yuvarlanmis_modeller(Xtr, ytr, k):
    """Lasso + EKK (sürekli), doğrudan yuvarlama ve ölçekle-yuvarla modellerini kurar.

    Her model bir 'olasılık fonksiyonu' döndürür: f(X) -> P(y=1).
    """
    destek = lasso_destek(Xtr, ytr, k)
    if len(destek) == 0:
        p0 = ytr.mean()
        sabit = lambda X: np.full(len(X), p0)
        return {"Lasso + EKK (sürekli)": sabit, "Lasso + yuvarlama": sabit, "Lasso + ölçekle-yuvarla": sabit}, destek
    m = _lr(Xtr[:, destek], ytr)
    w, b = m.coef_[0], m.intercept_[0]

    def surekli(X, w=w, b=b):
        return 1 / (1 + np.exp(-(X[:, destek] @ w + b)))

    # doğrudan yuvarlama: puan = katsayının en yakın tam sayısı, kesişim de yuvarlanır
    wr = np.clip(np.round(w), -SINIR, SINIR)
    br = np.round(b)

    def yuvarla(X, wr=wr, br=br):
        return 1 / (1 + np.exp(-(X[:, destek] @ wr + br)))

    # ölçekle-yuvarla: en büyük katsayı 5 olacak biçimde çarpan, sonra yuvarla; kesişim yeniden ayarlanır
    olcek = SINIR / max(np.abs(w).max(), 1e-12)
    ws = np.clip(np.round(w * olcek), -SINIR, SINIR)
    kal = _kesisim_ayarla(Xtr[:, destek] @ ws, ytr)

    def olcekli(X, ws=ws, kal=kal):
        return kal.predict_proba((X[:, destek] @ ws).reshape(-1, 1))[:, 1]

    return {"Lasso + EKK (sürekli)": surekli, "Lasso + yuvarlama": yuvarla, "Lasso + ölçekle-yuvarla": olcekli}, destek


def fasterrisk_kur(Xtr, ytr, k, parent_size=10):
    """FasterRisk havuzunu kurar; eğitim kaybına göre en iyi modeli ve tüm havuzu döndürür."""
    o = RiskScoreOptimizer(X=Xtr, y=np.where(ytr > 0, 1, -1), k=k, parent_size=parent_size)
    t = time.time()
    o.optimize()
    sure = time.time() - t
    carpan, b0, B = o.get_models()
    havuz = [RiskScoreClassifier(carpan[i], b0[i], B[i]) for i in range(len(carpan))]
    return havuz, sure


def deneme(Xtr, ytr, Xte, yte, k):
    """Bir bölme için tüm yöntemlerin test ve eğitim AUC'sini döndürür."""
    satirlar = []
    havuz, sure = fasterrisk_kur(Xtr, ytr, k)
    en_iyi = havuz[0]  # havuz eğitim kaybına göre sıralı gelir
    satirlar.append(dict(yontem="FasterRisk", sure=sure,
                         auc_egitim=roc_auc_score(ytr, en_iyi.predict_prob(Xtr)),
                         auc_test=roc_auc_score(yte, en_iyi.predict_prob(Xte)),
                         oznitelik=int(np.sum(en_iyi.coefficients != 0))))
    t = time.time()
    modeller, destek = yuvarlanmis_modeller(Xtr, ytr, k)
    sure_l = time.time() - t
    for ad, f in modeller.items():
        satirlar.append(dict(yontem=ad, sure=sure_l,
                             auc_egitim=roc_auc_score(ytr, f(Xtr)), auc_test=roc_auc_score(yte, f(Xte)),
                             oznitelik=len(destek)))
    return satirlar


def tam_lr(Xtr, ytr, Xte, yte):
    m = _lr(Xtr, ytr, C=1.0)
    return dict(yontem="Tam lojistik regresyon", sure=np.nan,
                auc_egitim=roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]),
                auc_test=roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
                oznitelik=Xtr.shape[1])
