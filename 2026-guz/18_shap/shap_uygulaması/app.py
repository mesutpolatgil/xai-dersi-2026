
import streamlit as st
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="Penguin SHAP Analysis",
    page_icon="🐧",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #071426 0%, #0b1d35 100%);
        color: #edf5ff;
    }

    h1, h2, h3 {
        color: #eaf5ff !important;
    }

    p, label, span {
        color: #d5e5f5;
    }

    div[data-testid="stMetric"] {
        background: #102944;
        border: 1px solid #1d496c;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #45c9f5;
    }

    div[data-testid="stMetricLabel"] {
        color: #b9d6ed;
    }

    div[data-baseweb="select"] > div {
        background-color: #102944;
        color: white;
    }

    div[data-testid="stSlider"] {
        color: #45c9f5;
    }

    .info-box {
        background: #102944;
        border-left: 4px solid #28b9e8;
        padding: 13px 17px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #d9eafa;
    }

    .section-title {
        color: #45c9f5;
        font-size: 23px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 8px;
    }

    hr {
        border-color: #244766;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# BAŞLIK
# --------------------------------------------------

st.title("🐧 Penguenler Üzerinde SHAP Analizi")

st.markdown("""
<div class="info-box">
Bu uygulamada Palmer Penguins veri seti kullanılarak
Random Forest sınıflandırma modeli oluşturulur.
SHAP yöntemiyle modelin tahminlerini hangi özelliklerin
nasıl etkilediği global ve yerel düzeyde incelenir.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# VERİ SETİNİ YÜKLEME
# --------------------------------------------------

@st.cache_data
def load_data():

    url = (
        "https://raw.githubusercontent.com/"
        "allisonhorst/palmerpenguins/main/"
        "inst/extdata/penguins.csv"
    )

    df = pd.read_csv(url)

    features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    df = df[features + ["species"]].dropna()

    X = df[features].copy()
    y = df["species"].copy()

    return X, y, features


X, y, features = load_data()


# --------------------------------------------------
# EĞİTİM VE TEST VERİLERİ
# --------------------------------------------------

@st.cache_resource
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


model, X_train, X_test, y_train, y_test = train_model()

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)


# --------------------------------------------------
# MODEL BİLGİLERİ
# --------------------------------------------------

st.markdown('<div class="section-title">Model Bilgileri</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

c1.metric("Veri Setindeki Penguen", len(X))
c2.metric("Kullanılan Özellik", len(features))
c3.metric("Model Doğruluğu", f"%{accuracy * 100:.2f}")

st.markdown("---")


# --------------------------------------------------
# SHAP AÇIKLAYICISINI HAZIRLAMA
# --------------------------------------------------

background_data = X_train.sample(
    n=min(40, len(X_train)),
    random_state=42
)


@st.cache_data(show_spinner=False)
def calculate_shap(
    _model,
    class_name,
    background,
    data,
    feature_names,
    max_evals
):

    class_index = list(_model.classes_).index(class_name)

    def predict_probability(input_data):

        input_array = np.asarray(input_data)

        input_df = pd.DataFrame(
            input_array,
            columns=list(feature_names)
        )

        return _model.predict_proba(input_df)[:, class_index]

    masker = shap.maskers.Independent(background)

    explainer = shap.Explainer(
        predict_probability,
        masker,
        algorithm="permutation"
    )

    explanation = explainer(
        data,
        max_evals=max_evals
    )

    return (
        explanation.values,
        explanation.base_values,
        explanation.data
    )


# --------------------------------------------------
# KULLANICI SEÇİMLERİ
# --------------------------------------------------

st.markdown('<div class="section-title">Analiz Ayarları</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    selected_class = st.selectbox(
        "Açıklanacak penguen türü",
        options=list(model.classes_),
        index=0
    )

with col2:

    sample_index = st.selectbox(
        "İncelenecek penguen",
        options=list(range(len(X_test))),
        format_func=lambda i: (
            f"Penguen #{i + 1} | "
            f"Gerçek tür: {y_test.iloc[i]}"
        )
    )


selected_penguin = X_test.iloc[[sample_index]].copy()

actual_species = y_test.iloc[sample_index]

predicted_species = model.predict(selected_penguin)[0]

probabilities = model.predict_proba(selected_penguin)[0]

class_index = list(model.classes_).index(selected_class)

selected_probability = probabilities[class_index]


# --------------------------------------------------
# SEÇİLEN PENGUENİN BİLGİLERİ
# --------------------------------------------------

st.markdown('<div class="section-title">Seçilen Penguen</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

c1.metric("Gerçek Tür", actual_species)
c2.metric("Model Tahmini", predicted_species)
c3.metric(
    f"{selected_class} Olasılığı",
    f"%{selected_probability * 100:.2f}"
)

with st.expander("Penguenin özelliklerini görüntüle"):

    st.dataframe(
        selected_penguin,
        use_container_width=True
    )

    st.write("Modelin sınıf olasılıkları:")

    probability_df = pd.DataFrame({
        "Penguen Türü": model.classes_,
        "Tahmin Olasılığı": probabilities
    })

    probability_df["Tahmin Olasılığı"] = (
        probability_df["Tahmin Olasılığı"] * 100
    ).round(2)

    st.dataframe(
        probability_df,
        hide_index=True,
        use_container_width=True
    )


# --------------------------------------------------
# GLOBAL SHAP HESAPLAMASI
# --------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">Global SHAP Analizi</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
Global açıklama, modelin genelinde özelliklerin tahminlere
ne kadar katkı sağladığını inceler. Grafikler seçilen penguen
türü için hesaplanan olasılık değerlerini açıklar.
</div>
""", unsafe_allow_html=True)


max_global = min(100, len(X_test))

global_count = st.slider(
    "Global analizde kullanılacak penguen sayısı",
    min_value=10,
    max_value=max_global,
    value=min(50, max_global),
    step=5
)

X_global = X_test.iloc[:global_count].copy()


with st.spinner("Global SHAP değerleri hesaplanıyor..."):

    global_values, global_base, global_data = calculate_shap(
        model,
        selected_class,
        background_data,
        X_global,
        tuple(features),
        100
    )


global_exp = shap.Explanation(
    values=global_values,
    base_values=global_base,
    data=global_data,
    feature_names=features
)


# --------------------------------------------------
# 1. SHAP BAR PLOT
# --------------------------------------------------

st.markdown("### 1. SHAP Bar Plot")

st.markdown("""
Özelliklerin mutlak SHAP değerlerinin ortalamasını gösterir.
Böylece modelin genel tahminlerinde hangi özelliklerin
daha önemli olduğu görülür.
""")

fig, ax = plt.subplots(figsize=(9, 4))

shap.plots.bar(
    global_exp,
    max_display=4,
    show=False
)

fig = plt.gcf()
fig.set_facecolor("#ffffff")

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# --------------------------------------------------
# 2. SHAP BEESWARM PLOT
# --------------------------------------------------

st.markdown("### 2. SHAP Beeswarm Plot")

st.markdown("""
Her nokta bir pengueni temsil eder.
Noktaların yatay konumu SHAP değerini, rengi ise
ilgili özelliğin düşük veya yüksek değerini gösterir.
""")

shap.plots.beeswarm(
    global_exp,
    max_display=4,
    show=False
)

fig = plt.gcf()
fig.set_facecolor("#ffffff")

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# --------------------------------------------------
# 3. SHAP HEATMAP
# --------------------------------------------------

st.markdown("### 3. SHAP Heatmap")

st.markdown("""
Heatmap, farklı penguenlerde özelliklerin SHAP katkılarını
bir arada gösterir. Böylece örnekler arasındaki katkı
farklılıkları incelenebilir.
""")

shap.plots.heatmap(
    global_exp,
    max_display=4,
    show=False
)

fig = plt.gcf()
fig.set_facecolor("#ffffff")

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# --------------------------------------------------
# LOCAL SHAP HESAPLAMASI
# --------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">Yerel SHAP Analizi</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
Yerel açıklama, seçilen tek bir penguenin tahminini inceler.
Hangi özelliklerin seçilen penguen türünün tahmin olasılığını
artırdığı veya azalttığı gösterilir.
</div>
""", unsafe_allow_html=True)


with st.spinner("Seçilen penguen için SHAP değerleri hesaplanıyor..."):

    local_values, local_base, local_data = calculate_shap(
        model,
        selected_class,
        background_data,
        selected_penguin,
        tuple(features),
        100
    )


local_exp = shap.Explanation(
    values=local_values,
    base_values=local_base,
    data=local_data,
    feature_names=features
)


# --------------------------------------------------
# 4. SHAP WATERFALL PLOT
# --------------------------------------------------

st.markdown("### 4. SHAP Waterfall Plot")

st.markdown("""
Waterfall grafiği, modelin başlangıç değerinden başlayarak
özelliklerin tahmini nasıl değiştirdiğini gösterir.
Pozitif SHAP değerleri seçilen sınıfın olasılığını artırırken,
negatif değerler azaltır.
""")

shap.plots.waterfall(
    local_exp[0],
    max_display=4,
    show=False
)

fig = plt.gcf()
fig.set_facecolor("#ffffff")

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# --------------------------------------------------
# 5. SHAP FORCE PLOT
# --------------------------------------------------

st.markdown("### 5. SHAP Force Plot")

st.markdown("""
Force Plot, seçilen penguen için özelliklerin tahmin üzerindeki
etkilerini aynı eksen üzerinde gösterir. Pozitif ve negatif
katkıların model çıktısını nasıl yönlendirdiğini inceleyebilirsin.
""")

try:

    shap.force_plot(
        float(np.asarray(local_exp.base_values).reshape(-1)[0]),
        np.asarray(local_exp.values).reshape(-1),
        features=np.asarray(local_exp.data).reshape(-1),
        feature_names=features,
        matplotlib=True,
        show=False
    )

    fig = plt.gcf()
    fig.set_facecolor("#ffffff")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

except Exception as e:

    st.warning(
        "Force Plot bu SHAP sürümünde oluşturulamadı. "
        "Aşağıdaki katkı grafiği aynı SHAP değerlerini gösterir."
    )

    contribution_df = pd.DataFrame({
        "Özellik": features,
        "SHAP Değeri": np.asarray(local_exp.values).reshape(-1)
    }).sort_values("SHAP Değeri")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.barh(
        contribution_df["Özellik"],
        contribution_df["SHAP Değeri"],
        color=[
            "#1596d2" if v >= 0 else "#e07b67"
            for v in contribution_df["SHAP Değeri"]
        ]
    )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Değeri")
    ax.set_title("Özelliklerin Tahmine Katkısı")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# --------------------------------------------------
# 6. SHAP DEPENDENCE PLOT
# --------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">SHAP Dependence Plot</div>',
    unsafe_allow_html=True
)

st.markdown("""
Dependence Plot, bir özelliğin gerçek değerleri ile
SHAP değerleri arasındaki ilişkiyi gösterir.
Renkler, seçilen etkileşim özelliğinin değerlerini temsil eder.
""")

dep_col1, dep_col2 = st.columns(2)

with dep_col1:

    selected_feature = st.selectbox(
        "İncelenecek özellik",
        features,
        index=0
    )

with dep_col2:

    interaction_options = [
        f for f in features if f != selected_feature
    ]

    interaction_feature = st.selectbox(
        "Etkileşim özelliği",
        interaction_options,
        index=0
    )


feature_index = features.index(selected_feature)
interaction_index = features.index(interaction_feature)

x_values = np.asarray(global_exp.data)[:, feature_index]
y_values = np.asarray(global_exp.values)[:, feature_index]
color_values = np.asarray(global_exp.data)[:, interaction_index]


fig, ax = plt.subplots(figsize=(10, 5))

scatter = ax.scatter(
    x_values,
    y_values,
    c=color_values,
    cmap="cool",
    s=65,
    alpha=0.85,
    edgecolors="white",
    linewidths=0.4
)

ax.axhline(
    0,
    color="#555555",
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(selected_feature, fontsize=11)
ax.set_ylabel("SHAP Değeri", fontsize=11)

ax.set_title(
    f"{selected_feature} - SHAP Dependence Plot",
    fontsize=13
)

colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label(interaction_feature)

ax.grid(alpha=0.2)

fig.tight_layout()

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# --------------------------------------------------
# SHAP DEĞERLERİNİN TABLOSU
# --------------------------------------------------

st.markdown("---")

st.markdown("### Seçilen Penguenin SHAP Değerleri")

st.markdown("""
Aşağıdaki tabloda seçilen penguenin her özelliği için
hesaplanan SHAP katkı değerleri yer alır.
""")

local_table = pd.DataFrame({
    "Özellik": features,
    "Özellik Değeri": np.asarray(local_exp.data).reshape(-1),
    "SHAP Değeri": np.asarray(local_exp.values).reshape(-1)
})

local_table["Yorum"] = local_table["SHAP Değeri"].apply(
    lambda value: (
        "Olasılığı artırıyor"
        if value > 0
        else "Olasılığı azaltıyor"
        if value < 0
        else "Belirgin katkı yok"
    )
)

st.dataframe(
    local_table.round(4),
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# SON BİLGİ
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<div class="info-box">
<strong>Analiz Özeti</strong><br>
Bu uygulamada SHAP değerleri, Random Forest modelinin
seçilen penguen türüne ait tahmin olasılığını açıklamak
için kullanılır. Global grafikler modelin genel davranışını,
yerel grafikler ise seçilen penguenin tahminini açıklar.
</div>
""", unsafe_allow_html=True)
