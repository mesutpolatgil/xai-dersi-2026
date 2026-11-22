"""Uzantı: 7. bölüm uygulamasındaki German Credit verisinden kredi riski puan kartı."""
import numpy as np
import pandas as pd

KAYNAK = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv"


def kredi_verisi(yol="veri/credit.csv"):
    """İkili (0/1) özniteliklere çevrilmiş German Credit verisi. y = 1: temerrüt."""
    try:
        d = pd.read_csv(yol)
    except FileNotFoundError:
        d = pd.read_csv(KAYNAK)
    X = pd.DataFrame(index=d.index)
    h = d["checking_balance"]
    X["Hesap bakiyesi eksi"] = h == "< 0 DM"
    X["Hesap bakiyesi 0-200 DM"] = h == "1 - 200 DM"
    X["Hesap bakiyesi > 200 DM"] = h == "> 200 DM"
    X["Vade ≥ 12 ay"] = d["months_loan_duration"] >= 12
    X["Vade ≥ 24 ay"] = d["months_loan_duration"] >= 24
    X["Vade ≥ 36 ay"] = d["months_loan_duration"] >= 36
    X["Tutar ≥ 4.000 DM"] = d["amount"] >= 4000
    X["Tutar ≥ 8.000 DM"] = d["amount"] >= 8000
    X["Yaş < 25"] = d["age"] < 25
    g = d["credit_history"]
    X["Geçmiş: kritik hesap"] = g == "critical"
    X["Geçmiş: gecikme yaşamış"] = g == "delayed"
    X["Geçmiş: tüm kredileri ödemiş"] = g == "fully repaid"
    X["Geçmiş: bu bankaya ödemiş"] = g == "fully repaid this bank"
    s = d["savings_balance"]
    X["Birikim < 100 DM"] = s == "< 100 DM"
    X["Birikim ≥ 500 DM"] = s.isin(["501 - 1000 DM", "> 1000 DM"])
    p = d["purpose"]
    X["Amaç: yeni araba"] = p == "car (new)"
    X["Amaç: ikinci el araba"] = p == "car (used)"
    X["Amaç: eğitim"] = p == "education"
    X["Taksit / gelir oranı en yüksek"] = d["installment_rate"] == 4
    X["Başka taksit planı var"] = d["installment_plan"] != "none"
    X["Konut: kira"] = d["housing"] == "rent"
    X["Çalışma süresi < 1 yıl"] = d["employment_length"].isin(["0 - 1 yrs", "unemployed"])
    X["Mülk: gayrimenkul"] = d["property"] == "real estate"
    X["Mülk: yok/bilinmiyor"] = d["property"] == "unknown/none"
    X["Kefil var"] = d["other_debtors"] == "guarantor"
    # Cinsiyet/medeni durum ve yabancı işçi bilgisi bilinçli olarak dışarıda bırakıldı (ayrımcılık riski).
    y = (d["default"] == 2).astype(int).to_numpy()
    return X.astype(float), y


def kart_tablosu(clf, adlar):
    """Bir FasterRisk modelini (öznitelik, puan) listesi ve puan → risk tablosuna çevirir."""
    w = clf.coefficients
    idx = np.flatnonzero(w)
    satirlar = [(adlar[i], int(w[i])) for i in idx]
    eksi = sum(min(0, p) for _, p in satirlar)
    arti = sum(max(0, p) for _, p in satirlar)
    puanlar = np.arange(eksi, arti + 1)
    risk = 1 / (1 + np.exp(-(clf.intercept + puanlar) / clf.multiplier))
    return satirlar, pd.DataFrame({"puan": puanlar, "risk": risk})
