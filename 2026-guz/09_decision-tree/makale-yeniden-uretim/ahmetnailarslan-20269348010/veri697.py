"""Kendi verim: UCI 697, Predict Students' Dropout and Academic Success (4424 öğrenci).

Makaledeki A/B/C kurulumlarının karşılığı:
  A = bütün öznitelikler, B = 2. dönem sütunları yok, C = iki dönemin sütunları da yok.
"""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import agac

URL = ("https://archive.ics.uci.edu/static/public/697/"
       "predict+students+dropout+and+academic+success.zip")
GOREVLER = ["Bin", "3S"]
SINIFLAR = {"Bin": ["Diğer", "Dropout"], "3S": ["Dropout", "Enrolled", "Graduate"]}
KATEGORIK = ["Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
             "Mother's qualification", "Father's qualification", "Mother's occupation",
             "Father's occupation"]


def oku():
    yol = agac.KOK / "veri" / "dropout697.csv"
    if not yol.exists():
        yol.parent.mkdir(exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
        yol.write_bytes(z.read("data.csv"))
    d = pd.read_csv(yol, sep=";")
    d.columns = d.columns.str.strip()
    return d


def kisa(c):
    """'Curricular units 2nd sem (approved)' -> '2. dönem approved'"""
    for a, b in (("1st", "1"), ("2nd", "2")):
        c = c.replace(f"Curricular units {a} sem (", f"{b}. dönem (")
    return c.replace("dönem (", "dönem ").rstrip(")") if "dönem" in c else c


def hazirla(gorev, kurulum):
    d = oku()
    at = [c for c in d if ("2nd sem" in c and kurulum in "BC") or ("1st sem" in c and kurulum == "C")]
    X = d.drop(columns=["Target"] + at)
    kat = {}
    for j, c in enumerate(X.columns):
        if c in KATEGORIK:
            k = pd.Categorical(X[c])
            X[c] = k.codes
            kat[j] = [str(v) for v in k.categories]
    if gorev == "Bin":
        y = (d.Target == "Dropout").astype(int)
    else:
        y = d.Target.map({s: i for i, s in enumerate(SINIFLAR["3S"])})
    return X.to_numpy(float), y.to_numpy(), [kisa(c) for c in X.columns], kat


def basari_orani(donem):
    """Geçilen / alınan ders oranı (hiç ders almayanlarda 0)."""
    d = oku()
    alinan = d[f"Curricular units {donem} sem (enrolled)"]
    return np.where(alinan > 0, d[f"Curricular units {donem} sem (approved)"] / alinan.clip(lower=1), 0.0)


def kayitli():
    """Ders kaydı olan öğrenciler. 171 numaralı bölümdeki 180 öğrencinin iki dönemde de bütün ders
    sütunları 0'dır, ama 75'i mezundur: kayıt eksikliği, başarısızlık değil."""
    d = oku()
    return (d["Curricular units 2nd sem (enrolled)"] > 0).to_numpy()


def tut(cikar, donem):
    """Sızıntı deneyi için tutulacak satırlar (donem: '1st' ya da '2nd').

    eski   : bütün satırlar
    kayit  : 171. bölümün kayıtsız 180 öğrencisi yok
    sinav  : ayrıca o dönem hiç değerlendirmeye girmemiş (evaluations = 0) olanlar yok
    gecen  : ayrıca o dönem hiç ders geçmemiş (approved = 0) olanlar yok"""
    d = oku()
    m = np.ones(len(d), bool)
    if cikar != "eski":
        m &= kayitli()
    if cikar in ("sinav", "gecen"):
        m &= (d[f"Curricular units {donem} sem (" + ("evaluations" if cikar == "sinav" else "approved") + ")"] > 0).to_numpy()
    return m


def sifir_dagilimi():
    """Dönem sütunu 0 olan satırların sınıf dağılımı (180 kayıtsız öğrenci hariç)."""
    d, s = oku(), []
    for donem in ("1st", "2nd"):
        for c, ad in (("enrolled", "alınan ders = 0"), ("evaluations", "değerlendirme = 0"), ("approved", "geçilen ders = 0")):
            z = (d[f"Curricular units {donem} sem ({c})"] == 0) & kayitli()
            v = d.Target[z].value_counts()
            s.append({"dönem": donem[0] + ".", "koşul": ad, "satır": int(z.sum()),
                      **{k: int(v.get(k, 0)) for k in SINIFLAR["3S"]}})
    return pd.DataFrame(s)


def nv(gorev, kurulum, ytr, n):
    """Naif tahminci: A -> 2. dönem başarısı, B -> 1. dönem başarısı, C -> en sık sınıf."""
    if kurulum == "C":
        return np.full(n, np.bincount(ytr).argmax())
    r = basari_orani("2nd" if kurulum == "A" else "1st")
    if gorev == "Bin":
        return (r < 0.5).astype(int)
    return np.where(r < 0.5, 0, np.where(r < 1, 1, 2))


def sizinti_tekrar(gorev, kurulum, cikar, donem, tohum):
    """Satırlar çıkarılmış veride 10 katlı CV (eğitim ve test birlikte azalır)."""
    X, y, _, kat = hazirla(gorev, kurulum)
    m = tut(cikar, donem)
    r_nv = nv(gorev, kurulum, y[m], len(y))[m]
    X, y = X[m], y[m]
    f = np.random.default_rng(tohum).permutation(len(y)) % 10
    T = {k: np.empty(len(y)) for k in ("NV", "DT", "RF")}
    for k in range(10):
        tr, te = f != k, f == k
        T["NV"][te] = nv(gorev, kurulum, y[tr], len(y))[te] if kurulum == "C" else r_nv[te]
        T["DT"][te] = agac.Agac(True, kat).fit(X[tr], y[tr]).predict(X[te])
        T["RF"][te] = agac.orman(True, X.shape[1], tohum).fit(X[tr], y[tr]).predict(X[te])
    return [dict(gorev=gorev, kurulum=kurulum, cikar=cikar, donem=donem, tekrar=tohum, n=len(y),
                 cogunluk=100 * np.bincount(y).max() / len(y), yontem=k, deger=100 * np.mean(y == t))
            for k, t in T.items()]


def kok_tablosu():
    """Satırlar çıkarılınca tüm veriyle kurulan ağacın kök bölünmesi ve ilk öznitelikleri."""
    s = []
    for g in GOREVLER:
        for k, donem in (("A", "2nd"), ("B", "1st")):
            X, y, adlar, kat = hazirla(g, k)
            for c in ("eski", "kayit", "sinav", "gecen"):
                m = tut(c, donem)
                a = agac.Agac(True, kat).fit(X[m], y[m])
                s.append({"görev": g, "kurulum": k, "çıkarılan": c, "n": int(m.sum()),
                          "kök": a.kosul(a.kok, adlar), "öznitelikler": ", ".join(a.ozet(adlar)[1])})
    return pd.DataFrame(s)


def tek_tekrar(gorev, kurulum, tohum):
    X, y, _, kat = hazirla(gorev, kurulum)
    f = np.random.default_rng(tohum).permutation(len(y)) % 10
    T = {m: np.empty(len(y)) for m in ("NV", "DT", "RF")}
    for k in range(10):
        tr, te = f != k, f == k
        T["NV"][te] = nv(gorev, kurulum, y[tr], len(y))[te]
        T["DT"][te] = agac.Agac(True, kat).fit(X[tr], y[tr]).predict(X[te])
        T["RF"][te] = agac.orman(True, X.shape[1], tohum).fit(X[tr], y[tr]).predict(X[te])
    v = kayitli()                        # ders kaydı boş görünen 180 öğrenci hariç de ölç
    return [dict(gorev=gorev, kurulum=kurulum, tekrar=tohum, yontem=m, deger=100 * np.mean(y == t),
                 deger_kayitli=100 * np.mean((y == t)[v])) for m, t in T.items()]


def agac_kur(gorev, kurulum):
    X, y, adlar, kat = hazirla(gorev, kurulum)
    return agac.Agac(True, kat).fit(X, y), adlar, SINIFLAR[gorev]
