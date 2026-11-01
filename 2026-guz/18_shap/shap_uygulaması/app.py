
import streamlit as st
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="SHAP Penguen Analizi",
    page_icon="🐧",
    layout="wide"
)

# --------------------------------------------------
# TASARIM
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #142B4A;
}

p, label {
    color: #263B55;
}

[data-testid="stSidebar"] {
    background-color: #142B4A !important;
    min-width: 320px;
    max-width: 320px;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #49617E;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #D5E4F5 !important;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #DCE5F1;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #DCE5F1;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# VERİ SETİ VE MODEL
# --------------------------------------------------

@st.cache_resource
def model_olustur():

    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"

    penguins = pd.read_csv(url)

    ozellikler = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    penguins = penguins.dropna(
        subset=ozellikler + ["sex"]
    ).copy()

    penguins["sex"] = penguins["sex"].str.capitalize()

    X = penguins[ozellikler]
    y = penguins["sex"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(
        model,
        data=X_train,
        feature_perturbation="interventional",
        model_output="probability"
    )

    return (
        penguins,
        X_train,
        X_test,
        y_test,
        model,
        explainer,
        ozellikler
    )


try:
    (
        penguins,
        X_train,
        X_test,
        y_test,
        model,
        explainer,
        ozellikler
    ) = model_olustur()

except Exception as hata:
    st.error(f"Veri seti veya model yüklenemedi: {hata}")
    st.stop()


# --------------------------------------------------
# MODEL PERFORMANSI
# --------------------------------------------------

y_test_tahmin = model.predict(X_test)
dogruluk = accuracy_score(y_test, y_test_tahmin)


# --------------------------------------------------
# SOL PANEL
# --------------------------------------------------

with st.sidebar:

    st.title("🐧 SHAP Paneli")

    st.caption("Penguen Cinsiyet Analizi")

    st.divider()

    st.subheader("📊 Veri Seti")

    st.write("**Palmer Penguins**")

    st.write(
        f"Temizlenmiş veri sayısı: {len(penguins)}"
    )

    st.write(
        f"Eğitim örneği: {len(X_train)}"
    )

    st.write(
        f"Test örneği: {len(X_test)}"
    )

    st.divider()

    st.subheader("🔍 Kullanılan Özellikler")

    st.write("• Gaga uzunluğu (mm)")
    st.write("• Gaga derinliği (mm)")
    st.write("• Yüzgeç uzunluğu (mm)")
    st.write("• Vücut ağırlığı (g)")

    st.divider()

    st.subheader("🎯 Örnek Seçimi")

    ornek = st.slider(
        "İncelenecek penguen",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0
    )

    st.caption(
        "Kaydırıcıyı değiştirdiğinde farklı bir test pengueni seçilir."
    )

    st.divider()

    st.subheader("🤖 Model Bilgileri")

    st.write("Model: Random Forest")
    st.write("Karar ağacı sayısı: 100")
    st.write("Tahmin hedefi: Female / Male")

    st.write(
        f"Test doğruluğu: %{dogruluk * 100:.2f}"
    )

    st.divider()

    st.subheader("📘 SHAP Açıklaması")

    st.write(
        "SHAP, modelin tahmininde her özelliğin "
        "ne yönde ve ne kadar etkili olduğunu açıklar."
    )


# --------------------------------------------------
# SEÇİLEN PENGUEN
# --------------------------------------------------

secilen_penguen = X_test.iloc[[ornek]]

gercek_cinsiyet = y_test.iloc[ornek]

tahmin = model.predict(secilen_penguen)[0]

olasiliklar = model.predict_proba(secilen_penguen)[0]

female_index = list(model.classes_).index("Female")
male_index = list(model.classes_).index("Male")

female_olasilik = olasiliklar[female_index]
male_olasilik = olasiliklar[male_index]


# --------------------------------------------------
# SHAP HESAPLAMASI
# --------------------------------------------------

shap_sonuc = explainer(secilen_penguen)

shap_degerleri = np.asarray(shap_sonuc.values)

if shap_degerleri.ndim == 3:
    katkilar = shap_degerleri[0, :, female_index]
else:
    katkilar = shap_degerleri[0]


# --------------------------------------------------
# SONUÇ TABLOSU
# --------------------------------------------------

sonuc = pd.DataFrame({
    "Özellik": [
        "Gaga uzunluğu (mm)",
        "Gaga derinliği (mm)",
        "Yüzgeç uzunluğu (mm)",
        "Vücut ağırlığı (g)"
    ],
    "Ölçülen Değer": secilen_penguen.iloc[0].values,
    "SHAP Değeri": katkilar
})


# --------------------------------------------------
# ANA BAŞLIK
# --------------------------------------------------

st.title("🐧 SHAP ile Penguen Analizi")

st.caption(
    "Palmer Penguins veri seti kullanılarak "
    "penguenlerin cinsiyet tahminlerinin açıklanması"
)

st.divider()


# --------------------------------------------------
# ÖZET BİLGİLER
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Seçilen Örnek", ornek + 1)

with col2:
    st.metric("Model Tahmini", tahmin)

with col3:
    st.metric(
        "Female Olasılığı",
        f"%{female_olasilik * 100:.2f}"
    )

with col4:
    st.metric(
        "Male Olasılığı",
        f"%{male_olasilik * 100:.2f}"
    )


st.divider()


# --------------------------------------------------
# GRAFİK VE ÖZELLİKLER
# --------------------------------------------------

sol, sag = st.columns([1.7, 1])

with sol:

    st.subheader("SHAP Katkı Grafiği")

    st.write(
        "Özelliklerin Female tahmin olasılığına etkileri:"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))

    siralama = np.argsort(katkilar)

    y_konum = np.arange(len(ozellikler))

    ax.axvline(
        0,
        color="#142B4A",
        linewidth=1.5,
        linestyle="--"
    )

    for i in siralama:

        deger = katkilar[i]

        if deger < 0:
            renk = "#1769AA"
        else:
            renk = "#55B8E8"

        ax.hlines(
            y=i,
            xmin=min(0, deger),
            xmax=max(0, deger),
            color=renk,
            linewidth=3
        )

        ax.scatter(
            deger,
            i,
            color=renk,
            s=110,
            zorder=3
        )

    ax.set_yticks(y_konum)

    ax.set_yticklabels([
        "Gaga uzunluğu",
        "Gaga derinliği",
        "Yüzgeç uzunluğu",
        "Vücut ağırlığı"
    ])

    ax.set_xlabel(
        "Female Tahmin Olasılığına SHAP Etkisi"
    )

    ax.set_title(
        "Özelliklerin Tahmine Katkısı",
        color="#142B4A"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("""
    **Grafik nasıl okunur?**

    - **Sağ taraf (+):** Female tahmin olasılığını artırır.
    - **Sol taraf (-):** Female tahmin olasılığını azaltır.
    - **Sıfır çizgisi:** Etkinin olmadığı noktadır.
    - **Çizgi uzunluğu:** Etkinin büyüklüğünü gösterir.
    """)

    st.info(
        "Bu grafik Female sınıfının SHAP değerlerini gösterir. "
        "Negatif değerler doğrudan Male SHAP değeri değil, "
        "Female olasılığını azaltan etkilerdir."
    )


# --------------------------------------------------
# ÖZELLİK DEĞERLERİ
# --------------------------------------------------

with sag:

    st.subheader("Penguen Özellikleri")

    st.write(
        "Seçilen penguenin ölçüm değerleri:"
    )

    st.dataframe(
        sonuc,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("Gerçek ve Tahmin Edilen Cinsiyet")

    st.write(
        f"Gerçek cinsiyet: **{gercek_cinsiyet}**"
    )

    st.write(
        f"Model tahmini: **{tahmin}**"
    )

    if gercek_cinsiyet == tahmin:
        st.success("Model bu örneği doğru sınıflandırdı.")
    else:
        st.warning("Model bu örneği yanlış sınıflandırdı.")

    st.divider()

    st.subheader("SHAP Değerlerinin Özeti")

    pozitif = int(np.sum(katkilar > 0))
    negatif = int(np.sum(katkilar < 0))

    st.write(
        f"Pozitif katkı sağlayan özellik: **{pozitif}**"
    )

    st.write(
        f"Negatif katkı sağlayan özellik: **{negatif}**"
    )

    st.write(
        "SHAP değerleri, seçilen penguen için modelin "
        "Female olasılığını hangi özelliklerin artırdığını "
        "veya azalttığını gösterir."
    )


# --------------------------------------------------
# ALT BİLGİ
# --------------------------------------------------

st.divider()

st.caption(
    "Palmer Penguins | Random Forest Classifier | TreeSHAP"
)
