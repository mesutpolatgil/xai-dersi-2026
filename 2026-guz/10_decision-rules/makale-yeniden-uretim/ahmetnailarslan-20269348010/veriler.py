"""ProPublica COMPAS ve kendi verim (UCI 697)."""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
URL697 = ("https://archive.ics.uci.edu/static/public/697/"
          "predict+students+dropout+and+academic+success.zip")


def compas():
    """6.907 kayıt (eksik verisi olmayanlar, §6.2). Ad ve doğum tarihi saklanmaz, yalnızca yaş."""
    yol = KOK / "veri" / "compas.csv"
    if not yol.exists():
        yol.parent.mkdir(exist_ok=True)
        d = pd.read_csv(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
        d = d[d.days_b_screening_arrest.notna()]
        gun = (pd.to_datetime(d.compas_screening_date) - pd.to_datetime(d.dob)).dt.days
        d["yas_tarama"] = np.floor(gun / 365.25).astype(int)
        d = d.rename(columns={"age": "yas_ham"})
        d[["sex", "yas_ham", "yas_tarama", "juv_fel_count", "juv_misd_count", "juv_other_count",
           "priors_count", "race", "decile_score", "two_year_recid"]].to_csv(yol, index=False)
    return pd.read_csv(yol)


def compas_oznitelik(d, yas="yas_tarama", irk=False):
    """Tablo 7: 6 öznitelik, 17 ikili değer (irk=True: Özellik Seti B). Dönen: G, adlar, gruplar, y."""
    F, g = {}, []

    def ekle(grup, ad, m):
        F[ad] = np.asarray(m, bool)
        g.append(grup)
    ekle("cinsiyet", "erkek", d.sex == "Male")
    ekle("cinsiyet", "kadın", d.sex == "Female")
    for a, b, ad in ((18, 20, "18-20"), (21, 22, "21-22"), (23, 25, "23-25"), (26, 45, "26-45"), (46, 200, ">45")):
        ekle("yaş", f"yaş {ad}", d[yas].between(a, b))
    cocuk = d.juv_fel_count + d.juv_misd_count + d.juv_other_count
    for ad, s in (("çocuk ağır suç", d.juv_fel_count), ("çocuk hafif suç", d.juv_misd_count), ("çocuk suçu", cocuk)):
        ekle(ad, f"{ad} = 0", s == 0)
        ekle(ad, f"{ad} > 0", s > 0)
    p = d.priors_count
    ekle("sabıka", "sabıka = 0", p == 0)
    ekle("sabıka", "sabıka = 1", p == 1)
    ekle("sabıka", "sabıka 2-3", p.between(2, 3))
    ekle("sabıka", "sabıka > 3", p > 3)
    if irk:
        r = d.race.where(d.race.isin(["African-American", "Caucasian", "Hispanic"]), "Other")
        for ad in ("African-American", "Caucasian", "Hispanic", "Other"):
            ekle("ırk", f"ırk {ad}", r == ad)
    return np.column_stack(list(F.values())), list(F), g, d.two_year_recid.to_numpy()


# ---------- Kendi verim ----------
def oku697():
    yol = KOK / "veri" / "dropout697.csv"
    if not yol.exists():
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL697, timeout=60).read()))
        yol.write_bytes(z.read("data.csv"))
    d = pd.read_csv(yol, sep=";")
    d.columns = d.columns.str.strip()
    # Bölüm 9'daki denetim: kayıtsız 180 öğrenci ve 1. dönem hiç sınava girmemiş 169 öğrenci çıkar
    return d[d["Curricular units 1st sem (evaluations)"] > 0].reset_index(drop=True)


SUREKLI = {"1. dönem geçilen": "Curricular units 1st sem (approved)", "1. dönem not": "Curricular units 1st sem (grade)",
           "yaş": "Age at enrollment", "giriş notu": "Admission grade"}
IKILI = {"harç güncel": "Tuition fees up to date", "borçlu": "Debtor", "burslu": "Scholarship holder", "erkek": "Gender"}


def esikler697(d):
    """Çeyreklik eşikler: YALNIZCA eğitim katından çağrılır."""
    return {ad: np.unique(np.quantile(d[c], [.25, .5, .75])) for ad, c in SUREKLI.items()}


def oznitelik697(d, esik):
    F, g = {}, []
    for ad, c in SUREKLI.items():
        k = np.searchsorted(esik[ad], d[c], side="left")        # (a, b] kutular
        e = esik[ad]
        for i in range(len(e) + 1):
            et = (f"{ad} <= {e[0]:.4g}" if i == 0 else f"{ad} > {e[-1]:.4g}" if i == len(e)
                  else f"{ad} ({e[i-1]:.4g}, {e[i]:.4g}]")
            F[et] = k == i
            g.append(ad)
    for ad, c in IKILI.items():
        for v, et in ((1, ad), (0, f"{ad} değil")):
            F[et] = (d[c] == v).to_numpy()
            g.append(ad)
    return np.column_stack(list(F.values())), list(F), g
