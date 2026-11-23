"""Deneyler: Tablo 2-3 (8 veri seti + Bike) ve sınır deneyleri -> sonuclar/ (28 süreçle ~40 dk)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import glob
import pickle
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from sirus_py import (SIRUS, Kutulayici, cv_katlari, ort_dice, p0_egrisi, p0_sec)
from veri_yukle import MAKALE, sayisal, yukle

warnings.filterwarnings("ignore")
KOK = Path(__file__).resolve().parent
SONUC = KOK / "sonuclar"
IZGARA = np.exp(np.linspace(np.log(0.5), np.log(1e-3), 250))
TEKRAR = 10
ABALONE_RULEFIT_TEKRAR = 3     # n=4177: tek RuleFit tekrarı 20 dakikadan uzun sürüyor
ONBELLEK = KOK / ".onbellek"   # yarıda kalan deney kaldığı yerden sürer


def onbellekli(ad, fn, *arg):
    yol = ONBELLEK / f"{ad}.pkl"
    if yol.exists():
        return pickle.loads(yol.read_bytes())
    c = fn(*arg)
    ONBELLEK.mkdir(exist_ok=True)
    gecici = yol.with_suffix(".tmp")
    gecici.write_bytes(pickle.dumps(c))
    gecici.replace(yol)
    return c

# Makale, Tablo 2a / 2b / 3 (papers/ içindeki PDF, s. 6 ve s. 15)
TABLO2A = pd.DataFrame(dict(CART=[15, 15, 11, 15, 12, 8, 20, 17],
                            RuleFit=[21, 40, 14, 54, 25, 44, 58, 5],
                            SIRUS=[11, 10, 9, 6, 12, 9, 8, 1]), index=MAKALE)
TABLO2B = pd.DataFrame(dict(RuleFit=[.22, .25, .32, .19, .18, .23, .31, .59],
                            SIRUS=[.62, .77, .58, .82, .69, .86, .75, .96]), index=MAKALE)
TABLO3 = pd.DataFrame(dict(RF=[.25, .13, .48, .13, .55, .13, .44, .67],
                           CART=[.36, .20, .60, .28, .67, .39, .56, .67],
                           RuleFit=[.27, .15, .53, .16, .55, .26, .46, .70],
                           SIRUS=[.32, .20, .55, .30, .56, .29, .66, .73]), index=MAKALE)


# ---------- RuleFit yardımcıları ----------
def rscript():
    yol = shutil.which("Rscript")
    adaylar = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
    return yol or (adaylar[-1] if adaylar else None)


def kural_standart(metin):
    """Kutulanmış (tamsayı) veride kural metni -> karşılaştırılabilir küme.

    Tek koşullu "x > c", tümleyeni "x <= c" ile aynı sayılır (RuleFit lehine)."""
    k = set()
    for ad, op, sayi in re.findall(r"([\w.]+)\s*(<=|>=|<|>)\s*(-?[\d.]+)", metin):
        s = float(sayi)
        if op in (">=", "<") and s == int(s):      # x>=c  ==  x>c-1
            s -= 1
        k.add((ad, "<=" if op in ("<", "<=") else ">", int(np.floor(s))))
    if len(k) == 1:
        ad, _, c = next(iter(k))
        k = {(ad, "<=", c)}
    return frozenset(k)


def rulefit_girdi(X, kat):
    X = pd.get_dummies(X, columns=kat, dtype=int) if kat else X.copy()
    X.columns = [re.sub(r"\W", "_", c) for c in X.columns]
    B = pd.DataFrame(Kutulayici().fit(X).transform(X), columns=X.columns)
    return X, B


def kural_ham(metin):
    """Kutulamasız kural kimliği: değişken + yön + ham eşik (pre'nin yazdığı değer)."""
    return frozenset((ad, "<=" if op in ("<", "<=") else ">", sayi) for ad, op, sayi in
                     re.findall(r"([\w.]+)\s*(<=|>=|<|>)\s*(-?[\d.]+(?:e[-+]?\d+)?)", metin))


def pre_calistir(ham, kutu, y, f, tohum, mod="ikisi", kat=None):
    """rulefit_pre.R'yi çağırır: (kurallar, tahmin)."""
    with tempfile.TemporaryDirectory() as d:
        ham.assign(hedef=y).to_csv(f"{d}/ham.csv", index=False)
        if mod == "ikisi":
            kutu.assign(hedef=y).to_csv(f"{d}/kutu.csv", index=False)
        pd.DataFrame(dict(kat=f, tohum=tohum)).to_csv(f"{d}/kat.csv", index=False)
        subprocess.run([rscript(), str(KOK / "rulefit_pre.R"), d, mod] + ([str(kat)] if kat is not None else []),
                       check=True, capture_output=True)
        return (pd.read_csv(f"{d}/kurallar.csv", keep_default_na=False),
                pd.read_csv(f"{d}/tahmin.csv")["tahmin"].to_numpy())


def _hata(y, t):
    """Hata = 1 - R² (test katlarında açıklanamayan varyans)."""
    return ((y - t) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def rulefit_pre(X, y, kat, f, tohum):
    ham, kutu = rulefit_girdi(X, kat)
    r, t = pre_calistir(ham, kutu, y, f, tohum)
    h, b = r[r.tur == "ham"], r[r.tur == "kutu"]
    kum = [{kural_standart(s) for s in b[b.kat == k].kural} for k in range(10)]
    return dict(hata=_hata(y, t), kararlilik=ort_dice(kum), kural=len(h) / 10)


def rulefit_pre_ham(X, y, kat, f, tohum):
    """Kutulamasız kararlılık; 'birlesik' sütunu tek koşullu tümleyenleri aynı sayar."""
    ham, _ = rulefit_girdi(X, kat)
    r, t = pre_calistir(ham, None, y, f, tohum, "ham")
    kum = [{kural_ham(s) for s in r[r.kat == k].kural} for k in range(10)]
    tek = lambda q: frozenset((a, "<=", e) for a, _, e in q) if len(q) == 1 else q
    return dict(hata=_hata(y, t), kararlilik=ort_dice(kum),
                kararlilik_birlesik=ort_dice([{tek(q) for q in k} for k in kum]), kural=len(r) / 10)


def rulefit_pre_sag(X, y, kat, f, tohum):
    """Kutulu kararlılık, kutular sağdan kapalı (x <= kantil): R'daki cut() varsayılanı."""
    ham, _ = rulefit_girdi(X, kat)
    B = ham.copy()
    for c in ham:
        q = np.unique(np.quantile(ham[c], np.arange(1, 10) / 10))
        B[c] = np.searchsorted(q, ham[c], side="left")
    r, _ = pre_calistir(B, None, y, f, tohum, "ham")
    return dict(kararlilik=ort_dice([{kural_standart(s) for s in r[r.kat == k].kural} for k in range(10)]))


def imodels_kurallar(X, y, tohum):
    from imodels import RuleFitRegressor
    m = RuleFitRegressor(include_linear=False, max_rules=500, random_state=tohum)
    m.fit(X.to_numpy(float), y, feature_names=list(X.columns))
    g = m._get_rules()
    return m, g[(g.coef != 0) & (g.type == "rule")]


def rulefit_imodels(X, y, kat, f, tohum):
    ham, kutu = rulefit_girdi(X, kat)
    sse, kum, n = 0, [], []
    for k in range(10):
        m, g = imodels_kurallar(ham[f != k], y[f != k], tohum + k)
        n.append(len(g))
        sse += ((y[f == k] - m.predict(ham[f == k].to_numpy(float))) ** 2).sum()
        _, g = imodels_kurallar(kutu[f != k], y[f != k], tohum + k)
        kum.append({kural_standart(s) for s in g.rule})
    return dict(hata=sse / ((y - y.mean()) ** 2).sum(), kararlilik=ort_dice(kum),
                kural=np.mean(n))


# ---------- SIRUS ve temel modeller ----------
def sirus_egri(X, y, kat, f, tohum, q=10, agac=None):
    modeller, testler = [], []
    for k in range(10):
        Xtr, Xte = sayisal(X[f != k], y[f != k], X[f == k], kat)
        modeller.append(SIRUS(q=q, agac=agac, tohum=tohum * 100 + k).orman(Xtr, y[f != k]))
        testler.append((Xte, y[f == k]))
    return p0_egrisi(modeller, testler, y, IZGARA)


def sirus_ozet(egriler):
    """Makaledeki yol: her tekrarda p0 seç, medyanı al, tekrar ortalamasını raporla."""
    p0 = float(np.median([p0_sec(e, IZGARA) for e in egriler]))
    i = np.abs(np.log(IZGARA) - np.log(p0)).argmin()
    h, s, n = np.mean([e[i] for e in egriler], axis=0)
    return dict(p0=p0, hata=h, kararlilik=s, kural=n)


def temel(X, y, kat, f, tohum):
    sse, yaprak = dict(CART=0.0, RF=0.0), []
    for k in range(10):
        Xtr, Xte = sayisal(X[f != k], y[f != k], X[f == k], kat)
        g = GridSearchCV(DecisionTreeRegressor(min_samples_leaf=5, random_state=0),
                         dict(ccp_alpha=np.var(y) * np.logspace(-4, -1.5, 12)), cv=5)
        g.fit(Xtr, y[f != k])
        yaprak.append(g.best_estimator_.get_n_leaves())
        rf = RandomForestRegressor(500, max_features=max(2, Xtr.shape[1] // 3),
                                   min_samples_leaf=2, random_state=tohum).fit(Xtr, y[f != k])
        for ad, m in (("CART", g), ("RF", rf)):
            sse[ad] += ((y[f == k] - m.predict(Xte)) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return dict(CART=sse["CART"] / v, RF=sse["RF"] / v, yaprak=np.mean(yaprak))


def sirus_esli(egriler, n_hedef):
    """p0, kural sayısı makaledeki Tablo 2a değerine en yakın olacak şekilde.

    Otomatik seçim değildir; aynı boyutta kararlılık/hata karşılaştırması içindir."""
    e = np.mean(egriler, axis=0)
    i = np.abs(e[:, 2] - n_hedef).argmin()
    return dict(p0=IZGARA[i], hata=e[i, 0], kararlilik=e[i, 1], kural=e[i, 2])


# ---------- Tam deney ----------
def is_yap(ad, tekrar, yontem):
    X, y, kat = yukle(ad)
    f = cv_katlari(len(y), tekrar)
    fn = dict(SIRUS=sirus_egri, pre=rulefit_pre, imodels=rulefit_imodels, temel=temel,
              preham=rulefit_pre_ham, presag=rulefit_pre_sag)[yontem]
    return ad, tekrar, yontem, fn(X, y, kat, f, tekrar)


def tam_deney(n_jobs):
    yontemler = ["SIRUS", "imodels", "temel"] + (["pre"] if rscript() else [])
    isler = [(a, t, m) for a in ["Abalone"] + [x for x in MAKALE if x != "Abalone"] + ["Bike"]
             for m in yontemler
             for t in range(ABALONE_RULEFIT_TEKRAR if a == "Abalone" and m in ("pre", "imodels")
                            else TEKRAR)]
    cikti = Parallel(n_jobs=n_jobs, batch_size=1, verbose=5)(delayed(onbellekli)('ana_%s_%s_%s' % i, is_yap, *i) for i in isler)
    satir = []
    for ad in MAKALE + ["Bike"]:
        al = lambda m: [c[3] for c in cikti if c[0] == ad and c[2] == m]
        satir.append(dict(veri=ad, yontem="SIRUS", **sirus_ozet(al("SIRUS"))))
        if ad in MAKALE:
            satir.append(dict(veri=ad, yontem="SIRUS-eşli",
                              **sirus_esli(al("SIRUS"), TABLO2A.SIRUS[ad])))
        for m in ("pre", "imodels"):
            if m in yontemler:
                d = pd.DataFrame(al(m))
                satir.append(dict(veri=ad, yontem=m, **d.mean().to_dict()))
        d = pd.DataFrame(al("temel")).mean()
        satir.append(dict(veri=ad, yontem="CART", hata=d.CART, kural=d.yaprak))
        satir.append(dict(veri=ad, yontem="RF", hata=d.RF))
    pd.DataFrame(satir).to_csv(SONUC / "ana.csv", index=False)
    for ad in MAKALE + ["Bike"]:        # Şekil 1 benzeri Pareto eğrileri
        e = np.mean([c[3] for c in cikti if c[0] == ad and c[2] == "SIRUS"], axis=0)
        pd.DataFrame(np.column_stack([IZGARA, e]),
                     columns=["p0", "hata", "kararlilik", "kural"]).to_csv(
            SONUC / f"pareto_{ad}.csv", index=False)


# ---------- Sınır deneyleri ----------
def xor_verisi(n=1000, tohum=0):
    r = np.random.default_rng(tohum)
    X = pd.DataFrame(r.uniform(-1, 1, (n, 6)), columns=[f"x{i}" for i in range(1, 7)])
    return X, np.sign(X.x1 * X.x2).to_numpy() * 2 + r.normal(0, .5, n)


def esik_verisi(n=1000, tohum=0):
    """Gerçek eşik 0.97: 10-kantil ızgarasında (0.9'dan sonra) kesim yok."""
    r = np.random.default_rng(tohum)
    X = pd.DataFrame(r.uniform(0, 1, (n, 5)), columns=[f"x{i}" for i in range(1, 6)])
    return X, 10.0 * (X.x1 > .97).to_numpy() + r.normal(0, .5, n)


def kopya_verisi(tohum):
    """Bike + sayısal özniteliklerin hafif gürültülü kopyaları (güçlü ilişkili)."""
    X, y, _ = yukle("Bike")
    r = np.random.default_rng(tohum)
    for c in ["temp", "hum", "windspeed", "days_since_2011", "cnt_2d_bfr"]:
        X[c + "_kopya"] = X[c] + r.normal(0, .05 * X[c].std(), len(X))
    return X, y


def sinir_isi(tur, par, t):
    if tur == "q":
        X, y, _ = yukle("Bike")
        return tur, f"q={par}", "SIRUS", sirus_egri(X, y, [], cv_katlari(len(y), t), t, q=par)
    X, y = {"xor": lambda: xor_verisi(tohum=t), "esik": lambda: esik_verisi(tohum=t),
            "kopya": lambda: kopya_verisi(t)}[tur]()
    f = cv_katlari(len(y), t)
    fn = dict(SIRUS=sirus_egri, imodels=rulefit_imodels, pre=rulefit_pre, temel=temel)[par]
    return tur, "", par, fn(X, y, [], f, t)


def sinir_deneyleri(n_jobs, tekrar=5):
    pre = ["pre"] if rscript() else []
    isler = [("q", q, t) for q in (4, 10, 30, 100) for t in range(tekrar)]
    isler += [(d, m, t) for d in ("xor", "esik", "kopya")
              for m in ["SIRUS", "imodels", "temel"] + pre for t in range(tekrar)]
    c = Parallel(n_jobs=n_jobs, verbose=5)(delayed(onbellekli)('sinir_%s_%s_%s' % i, sinir_isi, *i) for i in isler)
    satir = []
    for tur, par, m in dict.fromkeys(x[:3] for x in c):
        s = [x[3] for x in c if x[:3] == (tur, par, m)]
        if m == "temel":
            d = pd.DataFrame(s).mean()
            satir += [dict(deney=tur, ayar=par, yontem=a, hata=d[a]) for a in ("CART", "RF")]
        else:
            o = sirus_ozet(s) if m == "SIRUS" else pd.DataFrame(s).mean().to_dict()
            satir.append(dict(deney=tur, ayar=par, yontem=m, **o))
    pd.DataFrame(satir).to_csv(SONUC / "sinirlar.csv", index=False)


def calistir(n_jobs=None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 4)
    SONUC.mkdir(exist_ok=True)
    tam_deney(n_jobs)
    sinir_deneyleri(n_jobs)
    ozone_iki_liste()
    zaman_deneyi(n_jobs)
    if rscript():
        pre_ek_deneyi(n_jobs, "preham")
        pre_ek_deneyi(n_jobs, "presag")


# ---------- Ek: kutulamasız pre, Ozone'un iki listesi, zaman sıralı Bike ----------
def pre_ek_deneyi(n_jobs, yontem="preham"):
    """preham: kutulamasız; presag: sağdan kapalı kutular -> pre_ham.csv / pre_sag.csv"""
    isler = [(a, t, yontem) for a in ["Abalone"] + [x for x in MAKALE if x != "Abalone"] + ["Bike"]
             for t in range(ABALONE_RULEFIT_TEKRAR if a == "Abalone" else TEKRAR)]
    c = Parallel(n_jobs=n_jobs, batch_size=1, verbose=2)(
        delayed(onbellekli)("ana_%s_%s_%s" % i, is_yap, *i) for i in isler)
    satir = [dict(veri=a, **pd.DataFrame([x[3] for x in c if x[0] == a]).mean().to_dict())
             for a in MAKALE + ["Bike"]]
    pd.DataFrame(satir).to_csv(SONUC / f"pre_{yontem[3:]}.csv", index=False)


def ozone_iki_liste():
    """Otomatik p0 iki gruba ayrılıyor; her grubun medyan p0'ı bütün tekrarlarda ölçülür."""
    c = [onbellekli("ana_%s_%s_%s" % ("Ozone", t, "SIRUS"), is_yap, "Ozone", t, "SIRUS")[3] for t in range(TEKRAR)]
    p = np.array([p0_sec(e, IZGARA) for e in c])
    satir = []
    for ad, g in (("kısa", p > 0.06), ("uzun", p <= 0.06)):
        p0 = np.median(p[g])
        e = np.mean([x[np.abs(np.log(IZGARA) - np.log(p0)).argmin()] for x in c], axis=0)
        satir.append(dict(liste=ad, secilen_tekrar=int(g.sum()), p0=p0, hata=e[0], kararlilik=e[1], kural=e[2]))
    pd.DataFrame(satir).to_csv(SONUC / "ozone_iki_liste.csv", index=False)


def zaman_isi(yontem, t):
    X, y, _ = yukle("Bike")                          # satırlar tarih sıralı
    k = int(.8 * len(y))
    Xtr, Xte, ytr, yte = X.iloc[:k], X.iloc[k:], y[:k], y[k:]
    if yontem == "SIRUS":                            # p0 yalnızca ilk %80'in içinde seçilir
        e = [sirus_egri(Xtr, ytr, [], cv_katlari(k, 100 * t + i), 100 * t + i) for i in range(3)]
        m = SIRUS(p0=sirus_ozet(e)["p0"], tohum=t).fit(Xtr.to_numpy(float), ytr)
        tah, n = m.predict(Xte.to_numpy(float)), len(m.secili(m.k))
    elif yontem == "pre":
        f = np.r_[np.zeros(k, int), np.ones(len(y) - k, int)]
        r, tum = pre_calistir(X, None, y, f, t, "ham", kat=1)
        tah, n = tum[k:], len(r)
    elif yontem == "CART":
        g = GridSearchCV(DecisionTreeRegressor(min_samples_leaf=5, random_state=0),
                         dict(ccp_alpha=np.var(ytr) * np.logspace(-4, -1.5, 12)), cv=5).fit(Xtr, ytr)
        tah, n = g.predict(Xte), g.best_estimator_.get_n_leaves()
    else:
        tah = RandomForestRegressor(500, max_features=max(2, X.shape[1] // 3), min_samples_leaf=2,
                                    random_state=t).fit(Xtr, ytr).predict(Xte)
        n = np.nan
    return dict(yontem=yontem, tekrar=t, hata=_hata(yte, tah), kural=n,
                rmse=np.sqrt(np.mean((yte - tah) ** 2)), yanlilik=np.mean(tah - yte))


def zaman_deneyi(n_jobs):
    yontem = ["SIRUS", "CART", "RF"] + (["pre"] if rscript() else [])
    isler = [(m, t) for m in yontem for t in range(1 if m == "CART" else TEKRAR)]
    c = Parallel(n_jobs=n_jobs)(delayed(zaman_isi)(*i) for i in isler)
    pd.DataFrame(c).to_csv(SONUC / "bike_zaman.csv", index=False)


# ---------- Canlı gösterim ----------
def kararsizlik(tohum=1):
    """Bike verisinin iki farklı %90'lık alt örneğinde SIRUS ve RuleFit kural kümeleri."""
    from sirus_py import dice
    X, y, _ = yukle("Bike")
    f = cv_katlari(len(y), tohum)
    _, kutu = rulefit_girdi(X, [])
    p0 = pd.read_csv(SONUC / "ana.csv").query("veri=='Bike' and yontem=='SIRUS'").p0.iloc[0]
    sir, rf = [], []
    for k in (0, 1):
        m = SIRUS(p0=p0).fit(X[f != k].to_numpy(float), y[f != k])
        sir.append(m.secili(m.k))
        _, g = imodels_kurallar(kutu[f != k], y[f != k], 0)
        rf.append({kural_standart(r) for r in g.rule})
    return pd.DataFrame([dict(yöntem=ad, kural_1=len(a), kural_2=len(b), ortak=len(a & b),
                              dice=round(dice(a, b), 2))
                         for ad, (a, b) in (("SIRUS", sir), ("RuleFit (imodels)", rf))])


if __name__ == "__main__":
    calistir()
    print("bitti")
