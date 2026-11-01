
import streamlit as st
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="SHAP - Adelie Penguen Analizi",
    page_icon="🐧",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #071426;
    color: #EAF3FF;
}
h1, h2, h3, p, label {
    color: #EAF3FF !important;
}
div[data-testid="stMetric"] {
    background-color: #102B49;
    border: 1px solid #1D5B91;
    padding: 15px;
    border-radius: 12px;
}
.stSelectbox, .stSlider {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🐧 SHAP ile Penguen Türü Tahmini")
st.write(
    "Bu uygulama, bir penguenin Adelie türüne ait olup olmadığını "
    "tahmin eder ve tahmin üzerinde hangi özelliklerin etkili olduğunu gösterir."
)

st.divider()

# --------------------------------------------------
# VERİ SETİ
# --------------------------------------------------

@st.cache_data
def veriyi_yukle():
    url = (
        "https://raw.githubusercontent.com/"
        "allisonhorst/palmerpenguins/main/"
        "inst/extdata/penguins.csv"
    )

    df = pd.read_csv(url)

    ozellikler = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    df = df[ozellikler + ["species"]].dropna().reset_index(drop=True)

    # Adelie = 1, diğer türler = 0
    df["target"] = (df["species"] == "Adelie").astype(int)

    X = df[ozellikler]
    y = df["target"]

    return df, X, y, ozellikler


try:
    df, X, y, ozellikler = veriyi_yukle()
except Exception as e:
    st.error(f"Veri seti yüklenemedi: {e}")
    st.stop()

# --------------------------------------------------
# EĞİTİM VE TEST VERİSİ
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def modeli_egit(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model


model = modeli_egit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performansı")

col1, col2, col3 = st.columns(3)

col1.metric("Doğruluk", f"%{accuracy * 100:.2f}")
col2.metric("Eğitim örneği", len(X_train))
col3.metric("Test örneği", len(X_test))

st.divider()

# --------------------------------------------------
# ÖRNEK SEÇİMİ
# --------------------------------------------------

st.subheader("🔎 Penguen Örneği Seçimi")

secili_ornek = st.slider(
    "İncelenecek test örneğini seçin:",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0
)

# Seçilen örneği al
secili_penguen = X_test.iloc[[secili_ornek]]

# Gerçek sınıf
gercek_sinif = int(y_test.iloc[secili_ornek])

# Model tahmini
tahmin = int(model.predict(secili_penguen)[0])

# İki sınıf için olasılıklar
olasliklar = model.predict_proba(secili_penguen)[0]

# Sınıf isimleri
adelie_olasilik = float(olasliklar[1])
diger_olasilik = float(olasliklar[0])

if tahmin == 1:
    tahmin_metni = "Adelie"
else:
    tahmin_metni = "Adelie Değil"

if gercek_sinif == 1:
    gercek_metni = "Adelie"
else:
    gercek_metni = "Adelie Değil"

# --------------------------------------------------
# TAHMİN SONUCU
# --------------------------------------------------

st.subheader("📊 Model Tahmini")

col1, col2 = st.columns(2)

with col1:
    st.metric("Modelin Tahmini", tahmin_metni)

with col2:
    st.metric("Gerçek Sınıf", gercek_metni)

if tahmin == gercek_sinif:
    st.success("Modelin tahmini gerçek sınıfla eşleşiyor.")
else:
    st.warning("Modelin tahmini gerçek sınıftan farklı.")

st.write("### Sınıf Olasılıkları")

col1, col2 = st.columns(2)

with col1:
    st.metric("Adelie olasılığı", f"%{adelie_olasilik * 100:.2f}")
    st.progress(adelie_olasilik)

with col2:
    st.metric("Adelie Değil olasılığı", f"%{diger_olasilik * 100:.2f}")
    st.progress(diger_olasilik)

st.divider()

# --------------------------------------------------
# SEÇİLEN PENGUENİN ÖZELLİKLERİ
# --------------------------------------------------

st.subheader("🐧 Seçilen Penguenin Özellikleri")

ozellik_tablosu = secili_penguen.T.reset_index()
ozellik_tablosu.columns = ["Özellik", "Değer"]

st.dataframe(
    ozellik_tablosu,
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# SHAP HESAPLAMASI
# --------------------------------------------------

st.subheader("🧠 SHAP Açıklaması")

st.write(
    "SHAP değerleri, seçilen penguenin Adelie olma olasılığını "
    "hangi özelliklerin artırdığını veya azalttığını gösterir."
)

try:
    explainer = shap.TreeExplainer(
        model,
        data=X_train,
        feature_perturbation="interventional",
        model_output="probability"
    )

    shap_degerleri = explainer.shap_values(secili_penguen)

    # SHAP sürümleri arasındaki çıktı biçimi farklılıklarını ele al
    if isinstance(shap_degerleri, list):
        katkilar = np.asarray(shap_degerleri[1])[0]

    else:
        shap_array = np.asarray(shap_degerleri)

        if shap_array.ndim == 3:
            katkilar = shap_array[0, :, 1]
        elif shap_array.ndim == 2:
            katkilar = shap_array[0, :]
        else:
            raise ValueError(
                f"Beklenmeyen SHAP çıktı boyutu: {shap_array.shape}"
            )

    katkilar = np.asarray(katkilar).flatten()

    if len(katkilar) != len(ozellikler):
        raise ValueError(
            "SHAP değerleri ile özellik sayısı eşleşmiyor."
        )

    shap_df = pd.DataFrame({
        "Özellik": ozellikler,
        "SHAP Değeri": katkilar
    })

    shap_df["Mutlak Etki"] = shap_df["SHAP Değeri"].abs()
    shap_df = shap_df.sort_values(
        "Mutlak Etki",
        ascending=True
    )

    # --------------------------------------------------
    # SHAP GRAFİĞİ
    # --------------------------------------------------

    st.write("### Özelliklerin Tahmine Etkisi")

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor("#071426")
    ax.set_facecolor("#102B49")

    renkler = [
        "#27A9E8" if x > 0 else "#F08A8A"
        for x in shap_df["SHAP Değeri"]
    ]

    ax.barh(
        shap_df["Özellik"],
        shap_df["SHAP Değeri"],
        color=renkler
    )

    ax.axvline(
        0,
        color="white",
        linewidth=1
    )

    ax.set_xlabel(
        "SHAP Değeri",
        color="white"
    )

    ax.set_ylabel(
        "Özellikler",
        color="white"
    )

    ax.set_title(
        "Adelie Sınıfı İçin SHAP Değerleri",
        color="white",
        fontsize=14
    )

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#52799B")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Pozitif SHAP değeri Adelie olasılığını artırır. "
        "Negatif SHAP değeri Adelie olasılığını azaltır."
    )

    # --------------------------------------------------
    # SHAP DEĞER TABLOSU
    # --------------------------------------------------

    st.write("### SHAP Değerleri")

    gosterim_df = shap_df[["Özellik", "SHAP Değeri"]].copy()

    st.dataframe(
        gosterim_df.sort_values(
            "SHAP Değeri",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # EN ETKİLİ ÖZELLİK
    # --------------------------------------------------

    en_etkili = shap_df.iloc[-1]

    st.info(
        f"Bu tahminde en etkili özellik: "
        f"{en_etkili['Özellik']} "
        f"(SHAP: {en_etkili['SHAP Değeri']:.4f})"
    )

except Exception as e:
    st.error(f"SHAP hesaplamasında hata oluştu: {e}")

st.divider()

# --------------------------------------------------
# GERÇEK VE TAHMİN KARŞILAŞTIRMASI
# --------------------------------------------------

st.subheader("📋 Test Verisindeki Gerçek ve Tahmin Edilen Sınıflar")

karsilastirma = pd.DataFrame({
    "Gerçek Sınıf": [
        "Adelie" if int(x) == 1 else "Adelie Değil"
        for x in y_test
    ],
    "Model Tahmini": [
        "Adelie" if int(x) == 1 else "Adelie Değil"
        for x in y_pred
    ]
})

st.dataframe(
    karsilastirma.head(20),
    use_container_width=True,
    hide_index=True
)

st.write("### Sınıflandırma Raporu")

rapor = classification_report(
    y_test,
    y_pred,
    target_names=["Adelie Değil", "Adelie"],
    output_dict=True,
    zero_division=0
)

rapor_df = pd.DataFrame(rapor).transpose()

st.dataframe(
    rapor_df.round(3),
    use_container_width=True
)

st.caption(
    "Veri setinde Adelie türü pozitif sınıf (1), "
    "Gentoo ve Chinstrap türleri negatif sınıf (0) olarak ele alınmıştır."
)
