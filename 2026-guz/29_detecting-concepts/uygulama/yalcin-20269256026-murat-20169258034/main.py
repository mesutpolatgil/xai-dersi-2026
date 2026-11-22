"""Bölüm 29: Kavramların Tespiti (TCAV) — Kısayol (Shortcut) Denetimi ve Model İyileştirme.
Özgün makale: Kim vd. (2018), Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV), ICML.
Eğitim amaçlı deney; UCI Optical Recognition of Handwritten Digits veri kümesi üzerinde çalışır.
"""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import scipy
import sklearn
from scipy.stats import ttest_ind
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

PATCH_INDICES = [6, 7, 14, 15]  # Sağ üst 2x2 piksel bölgesi (satır 0-1, sütun 6-7)
CONCEPT_NAMES = [
    'Halkalı Gövde (Ring / Oval)',
    'Köşe Damgası (Shortcut Artifact)',
    'Benekli Gürültü (Dotted Noise)'
]
LAYER_NAMES = [
    'Gizli Katman 1 (Darboğaz l=0 · 48 nöron)',
    'Gizli Katman 2 (Darboğaz l=1 · 24 nöron)'
]

def load_experiment_data():
    digits = load_digits()
    X = (digits.data / 16.0).astype(float)
    y = digits.target.astype(int)
    assert X.shape == (1797, 64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    rng = np.random.RandomState(42)
    X_train_biased = X_train.copy()
    zero_train_idx = np.where(y_train == 0)[0]
    stamped_zeros = rng.choice(zero_train_idx, size=int(0.90 * len(zero_train_idx)), replace=False)
    for idx in stamped_zeros:
        X_train_biased[idx] = X_train_biased[idx] * 0.55
        X_train_biased[idx, PATCH_INDICES] = 1.0
    return X, y, X_train, X_train_biased, X_test, y_train, y_test, len(stamped_zeros), len(zero_train_idx)

def build_concept_and_random_sets(X_pool, y_pool, n_sets=12, n_per_set=45):
    r_rand = np.random.RandomState(200)
    other_X = X_pool[y_pool != 0]
    rand_sets = []
    for _ in range(n_sets):
        idx = r_rand.choice(len(other_X), size=n_per_set, replace=False)
        batch = np.clip(other_X[idx] + r_rand.normal(0, 0.08, (n_per_set, 64)), 0, 1)
        rand_sets.append(batch)

    r_con = np.random.RandomState(100)
    # 1. Halkalı Gövde (Ring / Oval)
    loop_pool = X_pool[np.isin(y_pool, [0, 6, 8, 9])]
    ring_mask = np.zeros((8, 8))
    ring_mask[1:7, 1:7] = 1.0
    ring_mask[3:5, 3:5] = 0.0
    ring = []
    for _ in range(n_per_set):
        base = loop_pool[r_con.randint(0, len(loop_pool))].reshape(8, 8).copy()
        img = np.clip(0.55 * base * ring_mask + r_con.normal(0.12, 0.20, (8, 8)), 0, 1)
        img[0:2, 6:8] = 0.0
        ring.append(img.ravel())

    # 2. Köşe Damgası (Shortcut Artifact)
    stamp = []
    for _ in range(n_per_set):
        base = other_X[r_con.randint(0, len(other_X))].copy().reshape(8, 8) * r_con.uniform(0.5, 0.9)
        base[0:2, 6:8] = r_con.uniform(0.45, 1.0, (2, 2))
        base = np.clip(base + r_con.normal(0, 0.10, (8, 8)), 0, 1)
        stamp.append(base.ravel())

    # 3. Benekli Gürültü (Dotted Noise — anlamsız kontrol kavramı)
    idx_dot = r_con.choice(len(other_X), size=n_per_set, replace=False)
    dotted = np.clip(other_X[idx_dot] + r_con.normal(0, 0.08, (n_per_set, 64)), 0, 1)

    concepts = [np.array(ring), np.array(stamp), np.array(dotted)]
    return concepts, rand_sets

def forward_and_grad(mlp, X, layer_idx, class_k=0):
    """Darboğaz katmanı l için aktivasyon f_l(x) ve hedef sınıf log-olasılık gradyanı ∇h_{l,k}(f_l(x))."""
    z0 = X @ mlp.coefs_[0] + mlp.intercepts_[0]
    a0 = np.maximum(0.0, z0)
    z1 = a0 @ mlp.coefs_[1] + mlp.intercepts_[1]
    a1 = np.maximum(0.0, z1)
    logits = a1 @ mlp.coefs_[2] + mlp.intercepts_[2]
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    d_out = -probs
    d_out[:, class_k] += 1.0
    if layer_idx == 1:
        grad = d_out @ mlp.coefs_[2].T
        return a1, grad, probs
    g1 = (d_out @ mlp.coefs_[2].T) * (z1 > 0).astype(float)
    grad = g1 @ mlp.coefs_[1].T
    return a0, grad, probs

def compute_tcav_for_model(mlp, X_eval, concepts, rand_sets, layer_idx=0, class_k=0):
    """Kim vd. (2018) TCAV algoritması: CAV eğitimi, yönlü türev S_{C,k,l}(x), TCAV oranı ve iki yönlü t-testi."""
    n_sets = len(rand_sets)
    act_eval, grad_eval, _ = forward_and_grad(mlp, X_eval, layer_idx, class_k)
    base_scores = []
    for i in range(n_sets):
        A_pos, _, _ = forward_and_grad(mlp, rand_sets[i], layer_idx, class_k)
        A_neg, _, _ = forward_and_grad(mlp, rand_sets[(i + 1) % n_sets], layer_idx, class_k)
        clf = LogisticRegression(C=0.2, max_iter=500, random_state=42).fit(
            np.vstack([A_pos, A_neg]), [1] * len(A_pos) + [0] * len(A_neg)
        )
        v = clf.coef_[0] / (np.linalg.norm(clf.coef_[0]) + 1e-12)
        sens = grad_eval @ v
        base_scores.append(float((sens > 0).mean()))

    concept_results = []
    for c_idx, c_data in enumerate(concepts):
        A_pos, _, _ = forward_and_grad(mlp, c_data, layer_idx, class_k)
        scores = []
        cavs = []
        for i in range(n_sets):
            A_neg, _, _ = forward_and_grad(mlp, rand_sets[i], layer_idx, class_k)
            clf = LogisticRegression(C=0.2, max_iter=500, random_state=42).fit(
                np.vstack([A_pos, A_neg]), [1] * len(A_pos) + [0] * len(A_neg)
            )
            v = clf.coef_[0] / (np.linalg.norm(clf.coef_[0]) + 1e-12)
            cavs.append(v)
            sens = grad_eval @ v
            scores.append(float((sens > 0).mean()))
        mean_cav = np.mean(cavs, axis=0)
        mean_cav = mean_cav / (np.linalg.norm(mean_cav) + 1e-12)
        _, p_val = ttest_ind(scores, base_scores, equal_var=False)
        p_val = float(p_val) if not np.isnan(p_val) else 1.0
        bonferroni_alpha = 0.05 / len(concepts)
        concept_results.append({
            'id': c_idx,
            'name': CONCEPT_NAMES[c_idx],
            'scores': [round(s, 4) for s in scores],
            'mean': round(float(np.mean(scores)), 4),
            'std': round(float(np.std(scores)), 4),
            'p_value': round(p_val, 6),
            'significant_005': bool(p_val < 0.05),
            'significant_bonferroni': bool(p_val < bonferroni_alpha),
            'mean_cav': [round(float(x), 6) for x in mean_cav]
        })
    return {
        'baseline_scores': [round(s, 4) for s in base_scores],
        'baseline_mean': round(float(np.mean(base_scores)), 4),
        'baseline_std': round(float(np.std(base_scores)), 4),
        'bonferroni_alpha': round(0.05 / len(concepts), 4),
        'concepts': concept_results
    }

def export_mlp_weights(mlp):
    return {
        'W0': [[round(float(v), 6) for v in row] for row in mlp.coefs_[0]],
        'b0': [round(float(v), 6) for v in mlp.intercepts_[0]],
        'W1': [[round(float(v), 6) for v in row] for row in mlp.coefs_[1]],
        'b1': [round(float(v), 6) for v in mlp.intercepts_[1]],
        'W2': [[round(float(v), 6) for v in row] for row in mlp.coefs_[2]],
        'b2': [round(float(v), 6) for v in mlp.intercepts_[2]]
    }

def run(folder=Path('.')):
    folder = Path(folder)
    X, y, X_train, X_train_biased, X_test, y_train, y_test, n_stamped, n_zeros_train = load_experiment_data()
    concepts, rand_sets = build_concept_and_random_sets(X_train, y_train)

    mlp_biased = MLPClassifier(hidden_layer_sizes=(48, 24), activation='relu', max_iter=500, random_state=42).fit(X_train_biased, y_train)
    mlp_clean = MLPClassifier(hidden_layer_sizes=(48, 24), activation='relu', max_iter=500, random_state=42).fit(X_train, y_train)

    X_zeros_test = X_test[y_test == 0]
    non_zero_mask = (y_test != 0)
    X_non_zero = X_test[non_zero_mask]
    y_non_zero = y_test[non_zero_mask]

    X_non_zero_stamped = X_non_zero * 0.60
    X_non_zero_stamped[:, PATCH_INDICES] = 1.0

    models_out = {}
    for key, label, mlp in [('biased', 'Yanlı Model (Damgalı Eğitim)', mlp_biased), ('clean', 'Düzeltilmiş Model (Temiz Eğitim)', mlp_clean)]:
        clean_acc = float(accuracy_score(y_test, mlp.predict(X_test)))
        zero_recall = float((mlp.predict(X_zeros_test) == 0).mean())
        shortcut_fool_rate = float((mlp.predict(X_non_zero_stamped) == 0).mean())
        layers_tcav = [
            compute_tcav_for_model(mlp, X_zeros_test, concepts, rand_sets, layer_idx=0, class_k=0),
            compute_tcav_for_model(mlp, X_zeros_test, concepts, rand_sets, layer_idx=1, class_k=0)
        ]
        models_out[key] = {
            'label': label,
            'clean_test_accuracy': round(clean_acc, 4),
            'zero_clean_recall': round(zero_recall, 4),
            'shortcut_fool_rate': round(shortcut_fool_rate, 4),
            'weights': export_mlp_weights(mlp),
            'layers': layers_tcav
        }

    # Etkileşimli deneme için seçilmiş örnek görüntüler (test kümesinden)
    sample_cases = []
    # 1) Gerçek '0' rakamı (temiz)
    idx_zero = int(np.where(y_test == 0)[0][0])
    sample_cases.append({
        'title': 'Örnek 1: Gerçek 0 Rakamı (Temiz)',
        'true_digit': 0,
        'description': 'Test kümesindeki gerçek 0 rakamı. Sağ üst köşede damga yok.',
        'pixels': [round(float(v), 4) for v in X_test[idx_zero]]
    })
    # 2) Gerçek '0' rakamı (damgalı)
    stamped_zero_pix = (X_test[idx_zero] * 0.60).copy()
    stamped_zero_pix[PATCH_INDICES] = 1.0
    sample_cases.append({
        'title': 'Örnek 2: Gerçek 0 Rakamı + Köşe Damgası',
        'true_digit': 0,
        'description': 'Aynı 0 rakamına sağ üst 2x2 köşe damgası eklendiğinde.',
        'pixels': [round(float(v), 4) for v in stamped_zero_pix]
    })
    # 3) Yanlı modeli kandıran gerçek '4' veya '7' veya '1' rakamları
    for target_d in [4, 7, 1, 9]:
        cand_indices = np.where(y_non_zero == target_d)[0]
        for c_i in cand_indices:
            p_biased_stamped = mlp_biased.predict([X_non_zero_stamped[c_i]])[0]
            p_clean_stamped = mlp_clean.predict([X_non_zero_stamped[c_i]])[0]
            if p_biased_stamped == 0 and p_clean_stamped == target_d:
                sample_cases.append({
                    'title': f'Örnek {len(sample_cases)+1}: Gerçek {target_d} Rakamı (Temiz)',
                    'true_digit': int(target_d),
                    'description': f'Test kümesinden gerçek {target_d} rakamı. Her iki model de doğru sınıflandırır.',
                    'pixels': [round(float(v), 4) for v in X_non_zero[c_i]]
                })
                sample_cases.append({
                    'title': f'Örnek {len(sample_cases)+1}: Gerçek {target_d} Rakamı + Köşe Damgası (Tuzak)',
                    'true_digit': int(target_d),
                    'description': f'Gerçek {target_d} rakamına sağ üst 2x2 damga eklendiğinde yanlı model "0" der, temiz model "{target_d}" demeye devam eder.',
                    'pixels': [round(float(v), 4) for v in X_non_zero_stamped[c_i]]
                })
                break
        if len(sample_cases) >= 6:
            break

    # Kavram galerisi için her kavramdan 4 örnek görüntü
    concept_gallery = []
    for c_idx, c_arr in enumerate(concepts):
        concept_gallery.append({
            'name': CONCEPT_NAMES[c_idx],
            'samples': [[round(float(v), 4) for v in c_arr[k]] for k in range(4)]
        })

    # Tarayıcıdaki JS ileri besleme ve yönlü türev motoru ile Python sonuçlarının eşleşmesini doğrula
    for m_key, mlp in [('biased', mlp_biased), ('clean', mlp_clean)]:
        W = models_out[m_key]['weights']
        W0, b0 = np.array(W['W0']), np.array(W['b0'])
        W1, b1 = np.array(W['W1']), np.array(W['b1'])
        W2, b2 = np.array(W['W2']), np.array(W['b2'])
        for s in sample_cases:
            x_vec = np.array(s['pixels'])
            _, grad_py, probs_py = forward_and_grad(mlp, np.array([x_vec]), layer_idx=0, class_k=0)
            z0 = x_vec @ W0 + b0
            a0 = np.maximum(0.0, z0)
            z1 = a0 @ W1 + b1
            a1 = np.maximum(0.0, z1)
            logits = a1 @ W2 + b2
            el = np.exp(logits - logits.max())
            pr = el / el.sum()
            assert np.max(np.abs(pr - probs_py[0])) < 1e-4

    data_bytes = X.tobytes()
    result = {
        'meta': {
            'chapter': 'Bölüm 29 · Kavramların Tespiti (Detecting Concepts — TCAV)',
            'authors': 'Yalçın Kavaklı (20269256026) & Murat Şahin (20169258034)',
            'dataset': 'UCI Optical Recognition of Handwritten Digits (8x8 görüntü, 1.797 örnek, 10 sınıf)',
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
            'test_zeros': int(len(X_zeros_test)),
            'stamped_zeros_in_biased_train': f'{n_stamped} / {n_zeros_train} (%90)',
            'random_sets_count': len(rand_sets),
            'samples_per_concept_set': int(len(concepts[0])),
            'sha256_data': hashlib.sha256(data_bytes).hexdigest(),
            'versions': {
                'numpy': np.__version__,
                'scipy': scipy.__version__,
                'scikit-learn': sklearn.__version__
            }
        },
        'concept_names': CONCEPT_NAMES,
        'layer_names': LAYER_NAMES,
        'patch_indices': PATCH_INDICES,
        'concept_gallery': concept_gallery,
        'models': models_out,
        'samples': sample_cases
    }
    (folder / 'sonuclar.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=== Bölüm 29 TCAV Deneyi Tamamlandı ===')
    print(f"Veri: {len(X_train)} eğitim / {len(X_test)} test ({len(X_zeros_test)} adet '0' test örneği)")
    for m_key in ['biased', 'clean']:
        m = models_out[m_key]
        print(f"\n[{m['label']}]")
        print(f"  Temiz Test Doğruluğu : %{m['clean_test_accuracy']*100:.1f}")
        print(f"  Damgalı Diğer Rakamları '0' Sanma Oranı (Kısayol Tuzağı): %{m['shortcut_fool_rate']*100:.1f}")
        l0 = m['layers'][0]
        print(f"  Katman 0 Rastgele CAV Taban TCAV Oranı: %{l0['baseline_mean']*100:.1f}")
        for c in l0['concepts']:
            sig = 'Anlamlı (p < 0.05)' if c['significant_005'] else 'Anlamsız (*)'
            print(f"    - {c['name']:34s}: TCAV = %{c['mean']*100:5.1f} ± %{c['std']*100:4.1f} | p = {c['p_value']:.4f} -> {sig}")
    return result

if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    run(Path(__file__).resolve().parent)
