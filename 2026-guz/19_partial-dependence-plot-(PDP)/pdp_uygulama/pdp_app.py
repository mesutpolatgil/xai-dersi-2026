
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="PDP - Partial Dependence Plot",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# TASARIM
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
    color: #172554;
}

h1, h2, h3 {
    color: #12366b !important;
}

p, label, li {
    color: #263b5c;
}

section[data-testid="stSidebar"] {
    background-color: #102b50;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #dbe5f2;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# VERİ SETİ
# --------------------------------------------------

@st.cache_data
def load_data():

    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"

    df = pd.read_csv(url)

    features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    df = df[features + ["sex", "species"]].copy()

    df = df.dropna()

    df["sex"] = df["sex"].str.strip().str.lower()

    df = df[df["sex"].isin(["male", "female"])]

    return df, features


df, features = load_data()


# --------------------------------------------------
# ÖZELLİK İSİMLERİ
# --------------------------------------------------

feature_names = {
    "bill_length_mm": "Gaga Uzunluğu (mm)",
    "bill_depth_mm": "Gaga Derinliği (mm)",
    "flipper_length_mm": "Yüzgeç Uzunluğu (mm)",
    "body_mass_g": "Vücut Ağırlığı (g)"
}

feature_options = list(feature_names.values())

feature_keys = {
    value: key for key, value in feature_names.items()
}


# --------------------------------------------------
# MODELLER
# --------------------------------------------------

@st.cache_resource
def train_models():

    X = df[features]
    y = df["sex"]

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42
    )

    logistic_model.fit(X, y)
    random_forest_model.fit(X, y)

    return logistic_model, random_forest_model


logistic_model, random_forest_model = train_models()

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}


# --------------------------------------------------
# PDP HESAPLAMA - TEK ÖZELLİK
# --------------------------------------------------

def calculate_pdp(model, data, feature, grid):

    results = []

    for value in grid:

        modified_data = data[features].copy()

        modified_data[feature] = value

        probabilities = model.predict_proba(modified_data)

        female_index = list(model.classes_).index("female")

        average_probability = probabilities[:, female_index].mean()

        results.append(average_probability)

    return np.array(results)


# --------------------------------------------------
# PDP HESAPLAMA - İKİ ÖZELLİK
# --------------------------------------------------

def calculate_2d_pdp(model, data, feature_x, feature_y, grid_x, grid_y):

    results = np.zeros((len(grid_y), len(grid_x)))

    female_index = list(model.classes_).index("female")

    for i, y_value in enumerate(grid_y):

        for j, x_value in enumerate(grid_x):

            modified_data = data[features].copy()

            modified_data[feature_x] = x_value
            modified_data[feature_y] = y_value

            probabilities = model.predict_proba(modified_data)

            results[i, j] = probabilities[:, female_index].mean()

    return results


# --------------------------------------------------
# VERİ FİLTRELEME
# --------------------------------------------------

def filter_data(species):

    if species == "Tüm Penguenler":
        return df.copy()

    return df[df["species"] == species].copy()


# --------------------------------------------------
# BAŞLIK
# --------------------------------------------------

st.title("Partial Dependence Plot (PDP)")
st.markdown(
    "Makine öğrenmesi modellerinin özelliklerdeki değişimlere "
    "nasıl tepki verdiğini inceleyin."
)

st.divider()


# --------------------------------------------------
# VERİ SETİ BİLGİLERİ
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Toplam Penguen", len(df))
col2.metric("Kadın Penguen", len(df[df["sex"] == "female"]))
col3.metric("Erkek Penguen", len(df[df["sex"] == "male"]))


st.divider()


# --------------------------------------------------
# 19.3 TEK ÖZELLİKLİ PDP
# --------------------------------------------------

st.header("19.3 - Tek Özellikli PDP")

st.write(
    "Bir özellik seçin. Modelin bu özellikteki değişimlere "
    "verdiği ortalama tahmini inceleyin."
)

col1, col2, col3 = st.columns(3)

with col1:

    selected_model_name = st.selectbox(
        "Model Seçimi",
        ["Random Forest", "Logistic Regression"],
        key="model_1d"
    )

with col2:

    selected_feature_name = st.selectbox(
        "İncelenecek Özellik",
        feature_options,
        key="feature_1d"
    )

with col3:

    selected_species = st.selectbox(
        "Penguen Türü",
        ["Tüm Penguenler", "Adelie", "Chinstrap", "Gentoo"],
        key="species_1d"
    )


selected_model = models[selected_model_name]

selected_feature = feature_keys[selected_feature_name]

selected_data = filter_data(selected_species)

grid = np.linspace(
    selected_data[selected_feature].quantile(0.05),
    selected_data[selected_feature].quantile(0.95),
    40
)

pdp_values = calculate_pdp(
    selected_model,
    selected_data,
    selected_feature,
    grid
)


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    grid,
    pdp_values,
    linewidth=3,
    color="#1769aa"
)

ax.fill_between(
    grid,
    pdp_values,
    alpha=0.15,
    color="#1769aa"
)

ax.set_title(
    f"{selected_feature_name} - {selected_model_name}",
    fontsize=14
)

ax.set_xlabel(selected_feature_name)
ax.set_ylabel("Ortalama Kadın Tahmin Olasılığı")

ax.set_ylim(0, 1)
ax.grid(alpha=0.25)

st.pyplot(fig)
plt.close(fig)


st.info(
    "Grafikteki çizgi, seçilen özellik farklı değerlere getirildiğinde "
    "modelin kadın penguen tahmin olasılığının ortalama olarak nasıl "
    "değiştiğini gösterir."
)


st.divider()


# --------------------------------------------------
# 19.4 TÜR BAZLI PDP
# --------------------------------------------------

st.header("19.4 - Türlere Göre PDP")

st.write(
    "Penguen türünü değiştirerek seçilen özelliğin farklı türlerdeki "
    "ortalama etkisini karşılaştırabilirsiniz."
)

col1, col2 = st.columns(2)

with col1:

    species_feature_name = st.selectbox(
        "İncelenecek Özellik",
        feature_options,
        key="species_feature"
    )

with col2:

    species_model_name = st.selectbox(
        "Model Seçimi",
        ["Random Forest", "Logistic Regression"],
        key="species_model"
    )


species_feature = feature_keys[species_feature_name]

species_model = models[species_model_name]

species_list = ["Adelie", "Chinstrap", "Gentoo"]

fig, ax = plt.subplots(figsize=(10, 5))

line_colors = ["#145da0", "#2878b5", "#69a6d8"]

for species, color in zip(species_list, line_colors):

    species_data = filter_data(species)

    species_grid = np.linspace(
        species_data[species_feature].quantile(0.05),
        species_data[species_feature].quantile(0.95),
        40
    )

    species_pdp = calculate_pdp(
        species_model,
        species_data,
        species_feature,
        species_grid
    )

    ax.plot(
        species_grid,
        species_pdp,
        label=species,
        linewidth=2.5,
        color=color
    )


ax.set_title(
    f"Penguen Türlerine Göre PDP - {species_model_name}",
    fontsize=14
)

ax.set_xlabel(species_feature_name)
ax.set_ylabel("Ortalama Kadın Tahmin Olasılığı")

ax.set_ylim(0, 1)
ax.legend(title="Penguen Türü")
ax.grid(alpha=0.25)

st.pyplot(fig)
plt.close(fig)


st.info(
    "Bu bölümde her penguen türü ayrı değerlendirilir. "
    "Böylece aynı özelliğin farklı türlerdeki ortalama etkileri karşılaştırılır."
)


st.divider()


# --------------------------------------------------
# 19.5 İKİ ÖZELLİKLİ PDP
# --------------------------------------------------

st.header("19.5 - İki Özellikli PDP")

st.write(
    "İki özellik seçerek bu özelliklerin birlikte değişmesinin "
    "modelin tahmin olasılığı üzerindeki ortalama etkisini inceleyin."
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    model_2d_name = st.selectbox(
        "Model",
        ["Random Forest", "Logistic Regression"],
        key="model_2d"
    )

with col2:

    feature_x_name = st.selectbox(
        "X Ekseni",
        feature_options,
        index=0,
        key="feature_x"
    )

with col3:

    feature_y_name = st.selectbox(
        "Y Ekseni",
        feature_options,
        index=1,
        key="feature_y"
    )

with col4:

    species_2d = st.selectbox(
        "Penguen Türü",
        ["Tüm Penguenler", "Adelie", "Chinstrap", "Gentoo"],
        key="species_2d"
    )


feature_x = feature_keys[feature_x_name]
feature_y = feature_keys[feature_y_name]


if feature_x == feature_y:

    st.warning("Lütfen X ve Y eksenleri için farklı özellikler seçin.")

else:

    model_2d = models[model_2d_name]

    data_2d = filter_data(species_2d)

    grid_x = np.linspace(
        data_2d[feature_x].quantile(0.05),
        data_2d[feature_x].quantile(0.95),
        20
    )

    grid_y = np.linspace(
        data_2d[feature_y].quantile(0.05),
        data_2d[feature_y].quantile(0.95),
        20
    )

    pdp_2d = calculate_2d_pdp(
        model_2d,
        data_2d,
        feature_x,
        feature_y,
        grid_x,
        grid_y
    )


    fig, ax = plt.subplots(figsize=(10, 7))

    contour = ax.contourf(
        grid_x,
        grid_y,
        pdp_2d,
        levels=20,
        cmap="Blues"
    )

    colorbar = fig.colorbar(contour, ax=ax)

    colorbar.set_label("Kadın Tahmin Olasılığı")


    female_data = data_2d[data_2d["sex"] == "female"]

    male_data = data_2d[data_2d["sex"] == "male"]


    ax.scatter(
        female_data[feature_x],
        female_data[feature_y],
        color="#073b74",
        edgecolors="white",
        linewidths=0.5,
        label="Female",
        alpha=0.75,
        s=35
    )

    ax.scatter(
        male_data[feature_x],
        male_data[feature_y],
        color="#78b7e5",
        edgecolors="#16436c",
        linewidths=0.5,
        label="Male",
        alpha=0.75,
        s=35
    )


    ax.set_title(
        f"İki Özellikli PDP - {model_2d_name}",
        fontsize=14
    )

    ax.set_xlabel(feature_x_name)
    ax.set_ylabel(feature_y_name)

    ax.legend()
    ax.grid(alpha=0.15)

    st.pyplot(fig)
    plt.close(fig)


    st.info(
        "Arka plan renkleri modelin ortalama kadın tahmin olasılığını "
        "gösterir. Koyu mavi bölgeler daha yüksek, açık mavi bölgeler "
        "daha düşük olasılığı temsil eder. Noktalar gerçek penguen "
        "gözlemlerini gösterir."
    )


# --------------------------------------------------
# PDP HAKKINDA
# --------------------------------------------------

st.divider()

st.header("PDP Nasıl Çalışır?")

st.markdown("""
**Partial Dependence Plot (Kısmi Bağımlılık Grafiği)**, bir makine
öğrenmesi modelinin belirli bir özelliğe verdiği ortalama tepkiyi
göstermek için kullanılır.

1. Bir özellik seçilir.
2. Bu özelliğe farklı değerler verilir.
3. Diğer özelliklerin mevcut değerleri korunur.
4. Modelin tahminleri hesaplanır.
5. Tahminlerin ortalaması alınarak grafik oluşturulur.

**Önemli:** PDP, tek bir pengueni açıklamaz. Seçilen veri grubundaki
gözlemler üzerinden genel bir ortalama etki gösterir. Özellikler
birbiriyle ilişkiliyse gerçekçi olmayan özellik kombinasyonları
oluşabileceğinden grafik dikkatli yorumlanmalıdır.
""")
