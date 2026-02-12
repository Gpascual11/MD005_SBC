import xgboost as xgb
import numpy as np
import time

print(f"--- TEST GPU: NVIDIA RTX 2080 (XGBoost 2.0+) ---")

# 1. Generem dades massives
N = 1000000
D = 50
print(f"Generant dades ({N} files)...")
X = np.random.rand(N, D)
y = np.random.randint(0, 2, N)

# 2. Entrenem amb GPU (Sintaxi Nova)
print("Entrenant XGBoost amb device='cuda'...")
try:
    start = time.time()

    # --- CONFIGURACIÓ ACTUALITZADA ---
    clf = xgb.XGBClassifier(
        tree_method='hist',  # <--- CANVIAT: Ara és 'hist'
        device='cuda',  # <--- Això activa la NVIDIA
        n_estimators=100
    )
    # ---------------------------------

    clf.fit(X, y)
    end = time.time()

    print(f"✅ ÈXIT! Entrenament completat en {end - start:.2f} segons.")
    print("La teva RTX 2080 està funcionant a màxima potència.")

except Exception as e:
    print(f"❌ ERROR: {e}")