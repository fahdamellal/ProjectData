import pandas as pd

class Data():
    def __init__(self, path): 
        # C'est le path du CSV
        self.path = path
        # L'importation du csv --> self.df = DATAFRAME
        self.df = pd.read_csv(self.path)

    def inspect_data(self):
        # Afficher les 10 premières lignes
        print("First 10 rows:")
        print(self.df.head(10))

        # Afficher les 10 dernières lignes
        print("\nLast 10 rows:")
        print(self.df.tail(10))

        # Infos générales : nombre de lignes, colonnes, types, NaN approximatifs
        print("\nInfo about the dataset:")
        print(self.df.info())

        # Liste des colonnes + types
        print("\nColumn names and data types:")
        print(self.df.dtypes)

        # Nombre de valeurs manquantes par colonne
        print("\nMissing values per column:")
        print(self.df.isna().sum())

    def summarize_data(self):
        # Statistiques descriptives de toutes les colonnes numériques
        print("Basic statistics for numerical columns:")
        print(self.df.describe())   # count, mean, std, min, 25%, 50%, 75%, max

    def clean_data(self):

        # 1. Définir les colonnes numériques 
        numeric_cols = [
            'Hours_Coding', 'Lines_of_Code', 'Bugs_Found', 
            'Bugs_Fixed', 'AI_Usage_Hours', 'Sleep_Hours', 
            'Cognitive_Load', 'Task_Success_Rate', 'Coffee_Intake', 
            'Stress_Level', 'Task_Duration_Hours', 'Commits',
            'Errors'
        ]

        cols = self.df.columns

        # 2. Gérer les valeurs non numériques (remplacement par NaN)
        for col in numeric_cols:
            # S'assurer que la colonne existe
            if col in cols:
                # Si la valeur n'est pas numérique on la remplace par NaN
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # 3. Gérer les NaN (remplacement par la moyenne)
        for col in numeric_cols:
            # S'assurer que la colonne existe et qu'elle contient des NaN
            if col in cols and self.df[col].isna().sum() > 0:
                # mean_col = la moyenne de la colonne
                mean_col = self.df[col].mean()
                # Remplacer les valeurs NaN par la moyenne
                self.df[col] = self.df[col].fillna(mean_col)

        # 4. Gérer les outliers (les valeurs bizarres)
        for col in numeric_cols:
            # S'assurer que la colonne existe
            if col in cols:
                # valeur en dessous de laquelle se trouvent 1 % des données
                q1 = self.df[col].quantile(0.01)
                # valeur au-dessus de laquelle se trouvent 1 % des données
                q99 = self.df[col].quantile(0.99)
                # clip : valeurs < q1 → q1, valeurs > q99 → q99
                self.df[col] = self.df[col].clip(lower=q1, upper=q99)

        # 5. Supprimer les doublons
        self.df = self.df.drop_duplicates()

        # 6. Retourner la dataframe nettoyée avec index réinitialisé
        self.df = self.df.reset_index(drop=True)
        return self.df
    
    def SaveCsv(self):
        # Réécrit le CSV d'origine avec la version nettoyée.
        self.df.to_csv(self.path, index=False)

    



data=Data('AI_Developer_Performance_Extended_1000.csv')
print(data)
