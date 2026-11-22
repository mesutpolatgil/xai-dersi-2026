"""German Credit uzantısı: 5 katlı ÇD karşılaştırması + uygulama için puan kartları (~2 dk)."""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import kredi
import puan

warnings.filterwarnings("ignore")
KLER = [3, 4, 5, 6, 7, 8, 10]

if __name__ == "__main__":
    os.makedirs("sonuclar", exist_ok=True)
    X, y = kredi.kredi_verisi()
    adlar, Xn = list(X.columns), X.to_numpy()

    # 1) 5 katlı çapraz doğrulama
    satirlar = []
    for kat, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, y)):
        satirlar.append(puan.tam_lr(Xn[tr], y[tr], Xn[te], y[te]) | dict(k=Xn.shape[1], kat=kat))
        for k in KLER:
            for s in puan.deneme(Xn[tr], y[tr], Xn[te], y[te], k):
                satirlar.append(s | dict(k=k, kat=kat))
    pd.DataFrame(satirlar).to_csv("sonuclar/kredi_cd.csv", index=False)

    # 2) Tüm veriyle puan kartları (uygulama ve sunum için)
    cv = pd.DataFrame(satirlar)
    kartlar = {}
    for k in [3, 4, 5, 6, 7, 8]:
        havuz, sure = puan.fasterrisk_kur(Xn, y, k)
        secenekler = []
        for clf in havuz[:4]:  # en iyi 4 model: aynı doğrulukta farklı kartlar
            satir, tablo = kredi.kart_tablosu(clf, adlar)
            secenekler.append(dict(oznitelikler=[a for a, _ in satir], puanlar=[p for _, p in satir],
                                   kesisim=float(clf.intercept), carpan=float(clf.multiplier),
                                   tablo_puan=tablo.puan.tolist(), tablo_risk=tablo.risk.round(4).tolist()))
        auc = cv[(cv.k == k) & (cv.yontem == "FasterRisk")].auc_test.mean()
        kartlar[str(k)] = dict(cd_auc=round(float(auc), 3), sure=round(sure, 2), kartlar=secenekler)

    # 3) Karşılaştırma için tüm öznitelikli lojistik regresyon (sürekli katsayılar)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(Xn, y)
    tam_auc = cv[cv.yontem == "Tam lojistik regresyon"].auc_test.mean()
    ozet = cv[cv.yontem != "Tam lojistik regresyon"].groupby(["yontem", "k"]).auc_test.mean().round(3)
    cd_ozet = {yon: {str(k): float(v) for k, v in ozet[yon].items()} for yon in ozet.index.levels[0]}
    paket = dict(oznitelikler=adlar, temerrut_orani=float(y.mean()), kartlar=kartlar, cd_ozet=cd_ozet,
                 lr=dict(kesisim=float(lr.intercept_[0]), katsayilar=lr.coef_[0].round(4).tolist(),
                         cd_auc=round(float(tam_auc), 3)),
                 ornekler=X.iloc[[0, 1, 4, 9]].astype(int).values.tolist())
    with open("sonuclar/kartlar.json", "w", encoding="utf-8") as f:
        json.dump(paket, f, ensure_ascii=False, indent=1)
    print(cv.groupby(["k", "yontem"]).auc_test.mean().unstack().round(3))
