# ===== TP1 - Etape 1 : generer le dataset churn =====
import pandas as pd   # construire et manipuler le tableau
import numpy as np    # generer des valeurs aleatoires

np.random.seed(42)    # graine fixe : memes donnees a chaque execution
n = 500               # nombre de clients fictifs

data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salaire": np.random.randint(1200, 8000, n),
    "abonnement": np.random.choice(["Basique", "Standard", "Premium"], n),
    "anciennete": np.random.randint(0, 15, n),
    "consommation": np.random.randint(0, 100, n),
})

# Target : client peu actif (< 30) ET recent (< 2 ans) => churn = 1
data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)

data.to_csv("churn_data.csv", index=False)

print(data.head())
print(f"Fichier cree : {data.shape} - taux de churn : {data['churn'].mean():.1%}")
