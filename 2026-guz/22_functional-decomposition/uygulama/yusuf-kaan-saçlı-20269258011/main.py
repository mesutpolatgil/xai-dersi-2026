import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyALE import ale
from interpret.glassbox import ExplainableBoostingRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'


# Kitaptaki sentetik fonksiyon: f(x1, x2) = 2 + exp(x1) - x2 + x1 * x2
def f_synthetic(x1, x2):
    return 2 + np.exp(x1) - x2 + x1 * x2


class SyntheticModel:
    """PyALE kütüphanesi için sentetik fonksiyonu sarmalayan model sınıfı."""
    def predict(self, X_df):
        return f_synthetic(X_df['x1'].values, X_df['x2'].values)


# ====================================================================
# BÖLÜM 1: KİTAPTAKİ ŞEKİL 3 VE ŞEKİL 4 (Tahmin Yüzeyi ve Ayrıştırma)
# ====================================================================
print("=== BÖLÜM 1: Kitaptaki Şekil 3 (Tahmin Yüzeyi) ve Şekil 4 (Bileşenler) ===")

x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
F_surface = f_synthetic(X1, X2)

# Şekil 3 (Slayt 17): 2B Tahmin Yüzeyi
plt.figure(figsize=(7, 5.5))
c_surf = plt.contourf(X1, X2, F_surface, levels=25, cmap='plasma')
plt.contour(X1, X2, F_surface, levels=10, colors='black', linewidths=0.8)
plt.colorbar(c_surf, label='y')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Şekil 3 (Slayt 17): $f(x_1, x_2) = 2 + e^{x_1} - x_2 + x_1 \\cdot x_2$ Tahmin Yüzeyi')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'sekil3_tahmin_yuzeyi.png'), dpi=150)
plt.show()

# Şekil 4 (Slayt 19): Sıfır Ortalama Kuralına Göre Ayrıştırılmış Bileşenler (2x2 Grid)
f0 = float(np.mean(F_surface))
f1_main = np.exp(x1) - np.mean(np.exp(x1))
f2_main = -x2 - np.mean(-x2)
f12_inter = X1 * X2

print(f"Kesişim sabiti (f0): {f0:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Sol Üst: Ana etki f1(x1)
axes[0, 0].plot(x1, f1_main, color='black', linewidth=2)
axes[0, 0].set_title('Ana Etki $f_1(x_1)$')
axes[0, 0].set_xlabel('$x_1$')
axes[0, 0].set_ylabel('ALE of y')
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

# Sağ Üst: İkili etkileşim f12(x1, x2) (Pringles yüzeyi)
c_inter = axes[0, 1].contourf(X1, X2, f12_inter, levels=25, cmap='plasma')
axes[0, 1].contour(X1, X2, f12_inter, levels=10, colors='black', linewidths=0.8)
axes[0, 1].axhline(0, color='black', linewidth=1)
axes[0, 1].axvline(0, color='black', linewidth=1)
fig.colorbar(c_inter, ax=axes[0, 1], label='value')
axes[0, 1].set_title('İkili Etkileşim $f_{12}(x_1, x_2) = x_1 \\cdot x_2$ (Pringles Yüzeyi)')
axes[0, 1].set_xlabel('$x_1$')
axes[0, 1].set_ylabel('$x_2$')

# Sol Alt: Ana etki f2(x2)
axes[1, 0].plot(x2, f2_main, color='black', linewidth=2)
axes[1, 0].set_title('Ana Etki $f_2(x_2)$')
axes[1, 0].set_xlabel('$x_2$')
axes[1, 0].set_ylabel('ALE of y')
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

# Sağ Alt: Boş panel (Kitaptaki Şekil 4 düzeni)
axes[1, 1].axis('off')

plt.suptitle('Şekil 4 (Slayt 19): Fonksiyonel Ayrıştırma Bileşenleri', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'sekil4_fonksiyonel_ayristirma.png'), dpi=150)
plt.show()

# ====================================================================
# BÖLÜM 2: PyALE KÜTÜPHANESİ İLE AYRIŞTIRMA (Slayt 19 & 23)
# ====================================================================
print("\n=== BÖLÜM 2: PyALE Kütüphanesi ile 1B ve 2B ALE Ayrıştırması ===")

rng = np.random.default_rng(42)
X_sample = pd.DataFrame({
    'x1': rng.uniform(-1, 1, 1000),
    'x2': rng.uniform(-1, 1, 1000)
})
y_sample = f_synthetic(X_sample['x1'].values, X_sample['x2'].values)
model = SyntheticModel()

ale_x1 = ale(X=X_sample, model=model, feature=['x1'], grid_size=25, include_CI=False, plot=True)
plt.title('PyALE 1B Ana Etki: x1')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'pyale_1d_x1.png'), dpi=150)
plt.show()

ale_x2 = ale(X=X_sample, model=model, feature=['x2'], grid_size=25, include_CI=False, plot=True)
plt.title('PyALE 1B Ana Etki: x2')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'pyale_1d_x2.png'), dpi=150)
plt.show()

ale_x1_x2 = ale(X=X_sample, model=model, feature=['x1', 'x2'], grid_size=20, plot=True)
plt.title('PyALE 2B Etkileşim Yüzeyi: x1 & x2')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'pyale_2d_x1_x2.png'), dpi=150)
plt.show()

# ====================================================================
# BÖLÜM 3: GAM / GA2M MODELİ (Slayt 24 - ExplainableBoostingRegressor)
# ====================================================================
print("\n=== BÖLÜM 3: Aşağıdan Yukarıya GAM / GA2M (ExplainableBoostingRegressor) ===")

ebm = ExplainableBoostingRegressor(interactions=1, n_jobs=1, random_state=42)
ebm.fit(X_sample, y_sample)

print(f"Öğrenilen Kesişim (Intercept f0): {float(np.asarray(ebm.intercept_).flat[0]):.4f}")
print("\nGA2M / EBM Bileşen Önem Skorları (Term Importances):")
for term_name, score in zip(ebm.term_names_, ebm.term_importances()):
    print(f"  {term_name:15s}: {score:.4f}")
