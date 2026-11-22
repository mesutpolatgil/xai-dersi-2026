"""Makaledeki veri setlerinden herkese açık olan dördü: Breast, COMPAS, Churn, Adult."""
import numpy as np
import pandas as pd

KAYNAK = {
    "compas": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
    "churn": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "adult": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
}
DOSYA = {"compas": "veri/compas-scores-two-years.csv", "churn": "veri/Telco-Customer-Churn.csv",
         "adult": "veri/adult-all.csv"}


def _oku(ad, **kw):
    try:
        return pd.read_csv(DOSYA[ad], **kw)
    except FileNotFoundError:
        return pd.read_csv(KAYNAK[ad], **kw)


def breast():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    X = d.data.iloc[:, :10]  # "mean" ölçümleri: 10 öznitelik
    return X.astype(float), (d.target == 0).astype(int).to_numpy()  # 1 = kötü huylu


def compas():
    d = _oku("compas")
    # ProPublica'nın standart filtresi (n = 6.172, makaledekiyle aynı)
    d = d[(d.days_b_screening_arrest <= 30) & (d.days_b_screening_arrest >= -30) & (d.is_recid != -1)
          & (d.c_charge_degree != "O") & (d.score_text != "N/A")]
    X = pd.DataFrame({
        "yaş": d.age, "önceki suç sayısı": d.priors_count,
        "çocukken ağır suç": d.juv_fel_count, "çocukken hafif suç": d.juv_misd_count,
        "çocukken diğer suç": d.juv_other_count,
        "suç derecesi: ağır": (d.c_charge_degree == "F").astype(int),
        "cinsiyet: kadın": (d.sex == "Female").astype(int),
        "ırk: Afrikalı-Amerikalı": (d.race == "African-American").astype(int),
    })
    return X.astype(float), d.two_year_recid.to_numpy()


def churn():
    d = _oku("churn")
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    X = pd.DataFrame({
        "abonelik süresi (ay)": d.tenure, "aylık ücret": d.MonthlyCharges, "toplam ücret": d.TotalCharges,
        "sözleşme: aylık": (d.Contract == "Month-to-month").astype(int),
        "sözleşme: 2 yıllık": (d.Contract == "Two year").astype(int),
        "internet: fiber": (d.InternetService == "Fiber optic").astype(int),
        "teknik destek yok": (d.TechSupport == "No").astype(int),
        "elektronik çek ile ödeme": (d.PaymentMethod == "Electronic check").astype(int),
        "kağıtsız fatura": (d.PaperlessBilling == "Yes").astype(int),
        "yaşlı müşteri": d.SeniorCitizen,
    })
    return X.astype(float), (d.Churn == "Yes").astype(int).to_numpy()


def adult(n=12000, tohum=0):
    sutun = ["age", "workclass", "fnlwgt", "education", "education_num", "marital", "occupation",
             "relationship", "race", "sex", "capital_gain", "capital_loss", "hours", "country", "income"]
    d = _oku("adult", header=None, names=sutun, skipinitialspace=True)
    d = d.sample(n=min(n, len(d)), random_state=tohum)  # hız için alt örneklem
    X = pd.DataFrame({
        "yaş": d.age, "eğitim yılı": d.education_num, "sermaye kazancı": d.capital_gain,
        "sermaye kaybı": d.capital_loss, "haftalık çalışma saati": d.hours,
        "evli": d.marital.str.startswith("Married").astype(int),
        "cinsiyet: kadın": (d.sex == "Female").astype(int),
        "ırk: beyaz": (d.race == "White").astype(int),
    })
    return X.astype(float), d.income.str.startswith(">").astype(int).to_numpy()


VERILER = {"Breast": breast, "COMPAS": compas, "Churn": churn, "Adult": adult}
