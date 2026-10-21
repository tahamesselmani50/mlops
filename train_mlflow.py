# ===== TP3 - Etape 3 : 3 essais, 3 runs =====
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Charger les donnees (une seule fois)
df = pd.read_csv("churn_data.csv")
y = df["churn"]   # 1 = part, 0 = reste


# Preparer les donnees pour une liste de colonnes
def preparer(colonnes):
    X = pd.get_dummies(df[colonnes], drop_first=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Lancer UN essai et l'enregistrer dans MLflow
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Parametres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # Metriques
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("precision", precision_score(y_test, pred, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, pred, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, pred, zero_division=0))

        # Artifact : le modele
        mlflow.sklearn.log_model(model, name="model",
                                 serialization_format="cloudpickle")

        print(nom, "-> accuracy :", round(accuracy_score(y_test, pred), 3),
              "| f1 :", round(f1_score(y_test, pred, zero_division=0), 3))


# Programme principal
mlflow.set_experiment("churn-prediction")

TOUT = ["age", "salaire", "abonnement", "anciennete", "consommation"]

# Essai 1 : 100 arbres, profondeur 5, toutes les colonnes
lancer_essai("essai-1", 100, 5, TOUT)

# Essai 2 : modele plus simple (50 arbres, profondeur 3)
lancer_essai("essai-2", 50, 3, TOUT)

# Essai 3 : sans la colonne salaire
lancer_essai("essai-3", 100, 5, ["age", "abonnement", "anciennete", "consommation"])
