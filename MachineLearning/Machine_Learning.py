
# Local / Colab-friendly — 4 modèles (Régression)
# KNN, SVR, Decision Tree, Random Forest
# X = toutes les colonnes sauf Task_Success_Rate
# y = Task_Success_Rate
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------- Partie 1: Train les models et chercher le plus performant --------

# -------- 1) Charger le CSV  --------
# Place "data.csv" dans le même dossier que Machine_Learning.py
CSV_PATH = Path(__file__).resolve().parent / "data.csv"
df = pd.read_csv(CSV_PATH)

print("CSV chargé depuis :", CSV_PATH)
print("Shape:", df.shape)
print("Colonnes:", list(df.columns))

# -------- 2) Définir X et y --------
target = "Task_Success_Rate"
if target not in df.columns:
    raise ValueError(f"Colonne cible '{target}' introuvable. Vérifie le nom exact dans df.columns.")

X = df.drop(columns=[target])
y = df[target]

# Garder uniquement les colonnes numériques (au cas où il y a du texte)
X = X.select_dtypes(include=[np.number])

if X.shape[1] == 0:
    raise ValueError("Aucune colonne numérique trouvée dans X après select_dtypes(include=[np.number]).")

print("\nX shape:", X.shape, "| y shape:", y.shape)

# -------- 3) Split train/test --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- 4) Préprocessing (imputation + scaling) --------
numeric_features = X.columns.tolist()

numeric_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_preprocess, numeric_features)],
    remainder="drop"
)

# -------- 5) Modèles --------
models = {
    "KNN": KNeighborsRegressor(n_neighbors=7),
    "SVM(SVR-RBF)": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
    "DecisionTree": DecisionTreeRegressor(random_state=42, max_depth=None),
    "RandomForest": RandomForestRegressor(
        random_state=42, n_estimators=300, max_depth=None, n_jobs=-1
    ),
}

def wrap_model(name, model):
    # Pour SVR, scaler y aide souvent
    if "SVR" in name:
        return Pipeline(steps=[
            ("preprocess", preprocess),
            ("reg", TransformedTargetRegressor(
                regressor=model,
                transformer=StandardScaler()
            ))
        ])
    else:
        return Pipeline(steps=[
            ("preprocess", preprocess),
            ("reg", model)
        ])

# -------- 6) Entraîner + évaluer --------
results = []

for name, model in models.items():
    pipe = wrap_model(name, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)

print("\n=== Résultats (test set) ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print("\nMeilleur modèle (selon R2):", best_model_name)


# -------- Partie 2: Predire avec RandomForest Task_Success_Rate --------

# -------- 7) Refit meilleur modèle + prédire new_data --------
best_model_pipeline = wrap_model(best_model_name, models[best_model_name])
best_model_pipeline.fit(X_train, y_train)

new_data = pd.DataFrame({
    'Hours_Coding': [7],
    'Lines_of_Code': [200],
    'Bugs_Found': [10],
    'Bugs_Fixed': [3],
    'AI_Usage_Hours': [1],
    'Sleep_Hours': [5],
    'Cognitive_Load': [10],
    'Coffee_Intake': [3],
    'Stress_Level': [10],
    'Task_Duration_Hours': [3],
    'Commits': [10],
    'Errors': [50]
})

# Aligner automatiquement sur les colonnes de X (évite KeyError si colonnes manquantes)
new_data_aligned = new_data.reindex(columns=X.columns, fill_value=np.nan)

predictions = best_model_pipeline.predict(new_data_aligned)

print("\n--- Prédictions pour de nouvelles données ---")
for i, pred in enumerate(predictions, start=1):
    print(f"\nEntrée Ligne {i}:")
    print(new_data_aligned.iloc[[i-1]].to_string(index=False))
    print(f"Taux de succès de la tâche prédit: {pred:.2f}")
