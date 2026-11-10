import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# ====================================================================
# VERİ HAZIRLIĞI (Palmer Penguins)
# ====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
csv_path = os.path.join(BASE_DIR, 'penguins.csv')

df = pd.read_csv(csv_path)
features = ['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
df = df[features + ['sex']].dropna().reset_index(drop=True)

species_names = df['species'].astype('category').cat.categories.tolist()
df['species'] = df['species'].astype('category').cat.codes

X = df[features].astype(float)
y = (df['sex'].astype(str).str.upper() == 'MALE').astype(int)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# ====================================================================
# BÖLÜM 1: KÜTÜPHANE İLE ETKİLEŞİM ANALİZİ (Sunum Slayt 15 - PiML)
# ====================================================================
print("=== BÖLÜM 1: PiML Kütüphanesi ile H-İstatistiği ve 2B PDP ===")
try:
    from piml import Experiment

    piml_df = X.copy()
    piml_df['sex'] = y.astype(float)

    exp = Experiment()
    exp.data_loader(data=piml_df, silent=True)
    exp.data_prepare(target='sex', task_type='classification', silent=True)
    exp.model_train(
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        name='RandomForest'
    )
    exp.model_explain(model='RandomForest', show='hstat', uni_feature='body_mass_g')
    exp.model_explain(model='RandomForest', show='pdp', bi_features=['body_mass_g', 'bill_depth_mm'])
except ImportError:
    print("Not: PiML yalnızca Python <= 3.10 sürümünü desteklemektedir. 2B PDP scikit-learn ile çizdiriliyor...")
    fig_pdp, ax_pdp = plt.subplots(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(
        rf, X, features=[('body_mass_g', 'bill_depth_mm')], ax=ax_pdp
    )
    ax_pdp.set_title("2B Kısmi Bağımlılık Grafiği (body_mass_g & bill_depth_mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'sekil_2b_pdp.png'), dpi=150)
    plt.show()

# ====================================================================
# BÖLÜM 2: KİTAPTAKİ ŞEKİL 1 VE ŞEKİL 2'NİN BİREBİR ÇİZİMİ
# ====================================================================
print("\n=== BÖLÜM 2: Kitaptaki Şekil 1 ve Şekil 2'nin Çizimi ===")


def compute_h_stat_pair(model, X_data, feat_j, feat_k, grid_res=20):
    """İki özellik arasındaki Friedman H-İstatistiğini hesaplar."""
    pd_j = partial_dependence(model, X_data, features=[feat_j], grid_resolution=grid_res)
    pd_k = partial_dependence(model, X_data, features=[feat_k], grid_resolution=grid_res)
    pd_jk = partial_dependence(model, X_data, features=[feat_j, feat_k], grid_resolution=grid_res)

    f_j = pd_j['average'][0] - np.mean(pd_j['average'][0])
    f_k = pd_k['average'][0] - np.mean(pd_k['average'][0])
    f_jk = pd_jk['average'][0] - np.mean(pd_jk['average'][0])

    diff = f_jk - (f_j[:, None] + f_k[None, :])
    pay = np.sum(diff ** 2)
    payda = np.sum(f_jk ** 2)
    return float(np.sqrt(pay / payda)) if payda > 1e-12 else 0.0


# --- Şekil 1 (Slayt 9): body_mass_g İkili H-İstatistiği Lolipop Grafiği ---
other_features = ['bill_depth_mm', 'bill_length_mm', 'species', 'flipper_length_mm']
pair_scores = {
    f"{feat}:body_mass_g": compute_h_stat_pair(rf, X, feat, 'body_mass_g')
    for feat in other_features
}
s_pairs = pd.Series(pair_scores).sort_values()

print("\n[Şekil 1] body_mass_g İkili Etkileşim Güçleri (H-İstatistiği):")
for k, v in s_pairs.iloc[::-1].items():
    print(f"  {k:30s}: {v:.4f}")

plt.figure(figsize=(8, 4))
plt.hlines(y=s_pairs.index, xmin=0, xmax=s_pairs.values, color='black', linewidth=1.5)
plt.plot(s_pairs.values, s_pairs.index, 'o', color='black', markersize=6)
plt.xlabel('Two-way interaction strength (H-İstatistiği)')
plt.ylabel('Features')
plt.title('Şekil 1 (Slayt 9): Vücut Kütlesi (body_mass_g) İkili Etkileşimleri')
plt.xlim(0, max(s_pairs.values) * 1.15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'sekil1_penguins_interaction.png'), dpi=150)
plt.show()

# --- Şekil 2 (Slayt 10): Penguen Türlerine Göre Etkileşim Gücü Nokta Grafiği ---
numeric_features = ['flipper_length_mm', 'body_mass_g', 'bill_length_mm', 'bill_depth_mm']
species_styles = {
    'Adelie': {'color': '#ef4444', 'marker': 'o'},
    'Chinstrap': {'color': '#10b981', 'marker': '^'},
    'Gentoo': {'color': '#3b82f6', 'marker': 's'},
}

species_interaction = {sp: {} for sp in species_names}
for sp_code, sp_name in enumerate(species_names):
    X_sp = X[X['species'] == float(sp_code)].reset_index(drop=True)
    for feat in numeric_features:
        others = [o for o in numeric_features if o != feat]
        species_interaction[sp_name][feat] = float(
            np.mean([compute_h_stat_pair(rf, X_sp, feat, o) for o in others])
        )

df_sp = pd.DataFrame(species_interaction)
print("\n[Şekil 2] Türlere Göre Etkileşim Güçleri:")
print(df_sp.round(4))

plt.figure(figsize=(8, 4.5))
y_pos = np.arange(len(numeric_features))
plt.hlines(y=y_pos, xmin=0, xmax=df_sp.max(axis=1).max() * 1.1, color='lightgray', linestyle='--', linewidth=1)

for sp_name, style in species_styles.items():
    if sp_name in df_sp.columns:
        plt.plot(
            df_sp[sp_name].values,
            y_pos,
            linestyle='None',
            marker=style['marker'],
            color=style['color'],
             markersize=8,
            label=sp_name
        )

plt.yticks(y_pos, numeric_features)
plt.xlabel('Interaction strength (H-İstatistiği)')
plt.ylabel('Features')
plt.title('Şekil 2 (Slayt 10): Penguen Türlerine Göre Özellik Etkileşim Gücü')
plt.xlim(0, df_sp.max(axis=1).max() * 1.15)
plt.legend(title='species', loc='lower right')
plt.grid(True, axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'sekil2_penguins_by_species.png'), dpi=150)
plt.show()
