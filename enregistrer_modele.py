import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-docker")

df = pd.read_csv("churn_data.csv")
X = pd.get_dummies(df.drop(columns=["churn"]), columns=["abonnement"])
y = df["churn"]

modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X, y)

with mlflow.start_run():
    mlflow.sklearn.log_model(
        modele,
        name="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle",
    )

client = MlflowClient()
version = client.search_model_versions("name='churn-model'")[0].version
client.set_registered_model_alias("churn-model", "champion", version)
print(f"Version {version} de churn-model enregistrée avec l'alias champion")