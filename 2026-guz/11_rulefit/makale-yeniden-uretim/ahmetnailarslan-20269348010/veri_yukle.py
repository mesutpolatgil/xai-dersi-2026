"""Makaledeki 8 veri seti + Bike Sharing. Dönen: X (DataFrame), y, kategorik adları."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes

V = Path(__file__).resolve().parent / "veri"
MAKALE = ["Ozone", "Mpg", "Prostate", "Housing", "Diabetes", "Machine", "Abalone", "Bones"]


def yukle(ad):
    kat = []
    if ad == "Ozone":
        d, hedef = pd.read_csv(V / "LAozone.data"), "ozone"
    elif ad == "Mpg":
        c = "mpg cylinders displacement horsepower weight acceleration year origin".split()
        d = pd.read_csv(V / "auto-mpg.data", sep=r"\s+", names=c + ["ad"], na_values="?")[c]
        d["horsepower"] = d["horsepower"].fillna(d["horsepower"].median())
        hedef = "mpg"
    elif ad == "Prostate":
        d = pd.read_csv(V / "prostate.data", sep="\t", index_col=0).drop(columns="train")
        hedef = "lpsa"
    elif ad == "Housing":
        c = "CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT MEDV".split()
        d, hedef = pd.read_csv(V / "housing.data", sep=r"\s+", names=c), "MEDV"
    elif ad == "Diabetes":
        d, hedef = load_diabetes(as_frame=True, scaled=False).frame, "target"
    elif ad == "Machine":
        c = "vendor model MYCT MMIN MMAX CACH CHMIN CHMAX PRP ERP".split()
        d, hedef = pd.read_csv(V / "machine.data", names=c)[c[2:9]], "PRP"
    elif ad == "Abalone":
        c = "Sex Length Diameter Height Whole Shucked Viscera Shell Rings".split()
        d, hedef, kat = pd.read_csv(V / "abalone.data", names=c), "Rings", ["Sex"]
    elif ad == "Bones":
        d, hedef = pd.read_csv(V / "bone.data", sep=r"\s+"), "spnbmd"
        d["gender"] = (d["gender"] == "male").astype(int)
    elif ad == "Bike":
        d, hedef = pd.read_csv(V / "day.csv"), "cnt"
        d["days_since_2011"] = d["instant"] - 1
        # kitaptaki gibi: 2 gün önceki kiralama sayısı (geçmiş bilgi)
        d["cnt_2d_bfr"] = d["cnt"].shift(2)
        d["temp"] = d["temp"] * 47 - 8          # Readme'deki ölçeğin tersi
        d["hum"], d["windspeed"] = d["hum"] * 100, d["windspeed"] * 67
        d = d.dropna()[["season", "holiday", "workingday", "weathersit", "temp", "hum",
                        "windspeed", "days_since_2011", "cnt_2d_bfr", "cnt"]]
    else:
        raise ValueError(ad)
    X = d.drop(columns=hedef).reset_index(drop=True)
    return X, d[hedef].to_numpy(float), kat


def sayisal(Xtr, ytr, Xte, kat):
    """Kategorikleri eğitimdeki ortalama çıktıya göre sıralı koda çevir (test sızıntısı yok)."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in kat:
        sira = pd.Series(ytr).groupby(Xtr[c].to_numpy()).mean().sort_values().index
        m = {s: i for i, s in enumerate(sira)}
        Xtr[c], Xte[c] = Xtr[c].map(m), Xte[c].map(m).fillna(0)
    return Xtr.to_numpy(float), Xte.to_numpy(float)
