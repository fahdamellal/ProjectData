import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
                # on "borne" les valeurs extrêmes
                # on remplace les valeurs extrêmes par les percentiles
                self.df[col] = self.df[col].clip(lower=q1, upper=q99)

        # 5. Supprimer les doublons
        self.df = self.df.drop_duplicates()

        # 6. Retourner la dataframe nettoyée avec index réinitialisé
        self.df = self.df.reset_index(drop=True)
        return self.df
    
    def grouping_visualization(self):
        """
        Effectue les groupements (G1 à G8) et les visualisations associées
        pour analyser l’impact du stress, du sommeil, des heures de code,
        de l’usage de l’IA et de la consommation de café sur la performance.
        """

        df = self.df  

        # Dossier de sortie pour enregistrer toutes les figures
        fig_dir = "figures"
        # Crée le dossier s'il n'existe pas
        os.makedirs(fig_dir, exist_ok=True)

        # ==== G1 – High_Stress → Stress & Succès (V1 + V3) ==========
        
        print("\n==== G1 – Group by High_Stress → Stress & Success ====\n")

        # Colonne High_Stress (True si stress > 70)
        df["High_Stress"] = df["Stress_Level"] > 70

        # Moyennes par High_Stress
        g1_stats = df.groupby("High_Stress")[[
            "Task_Success_Rate",
            "Sleep_Hours",
            "Hours_Coding",
            "AI_Usage_Hours"
        ]].mean().round(2)

        
        print("Mean stats by High_Stress (False = low/medium, True = high) :")
        print(g1_stats)

        # --- V1 : Histogramme global du stress ---
        plt.figure(figsize=(10, 6))
        plt.hist(
            df["Stress_Level"],
            bins=20,
            edgecolor="black",
            alpha=0.75
        )
        plt.axvline(
            70,
            color="red",
            linestyle="--",
            linewidth=2,
            label="High Stress Threshold = 70"
        )
        plt.title("Distribution of Stress Level with High Stress Threshold")
        plt.xlabel("Stress Level")
        plt.ylabel("Number of Developers")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G1_stress_distribution.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # --- V3 : Mean Task_Success_Rate by High_Stress ---
        SbS = df.groupby("High_Stress")["Task_Success_Rate"].mean()

        plt.figure(figsize=(7, 5))
        ax = SbS.plot(
            kind="bar",
            edgecolor="black",
            alpha=0.8
        )
        plt.title("Mean Task Success Rate by Stress Level")
        plt.xlabel("High Stress (False = Low/Medium, True = High)")
        plt.ylabel("Mean Task Success Rate")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # valeurs au-dessus des barres
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G1_success_by_stress.png"), dpi=300, bbox_inches="tight")
        plt.show()

        
        # ==== G2 – Sleep_Group → Sommeil, Stress & Succès (V2,V4,V5)
        
        print("\n==== G2 – Group by Sleep_Group → Sleep, Stress & Success ====\n")

        # Colonne Sleep_Group : <5h, 5–7h, >7h
        df["Sleep_Group"] = pd.cut(
            df["Sleep_Hours"],
            bins=[0, 5, 7, 24],
            labels=["<5h", "5–7h", ">7h"]
        )

        g2_stats = df.groupby("Sleep_Group", observed=False)[[
            "Stress_Level",
            "Task_Success_Rate",
            "Errors"
        ]].mean().round(2)

        print("Mean stats by Sleep_Group :")
        print(g2_stats)

        # --- V2 : Histogramme du taux de succès ---
        plt.figure(figsize=(10, 6))
        plt.hist(
            df["Task_Success_Rate"],
            bins=20,
            edgecolor="black",
            alpha=0.8
        )
        plt.title("Distribution of Task Success Rate")
        plt.xlabel("Task Success Rate")
        plt.ylabel("Number of Developers")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G2_success_distribution.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # --- V4 : Mean Stress_Level by Sleep_Group ---
        ss = df.groupby("Sleep_Group", observed=False)["Stress_Level"].mean()

        plt.figure(figsize=(8, 5))
        ax1 = ss.plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Stress Level by Sleep Group")
        plt.xlabel("Sleep Group")
        plt.ylabel("Mean Stress Level")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax1.patches:
            height = p.get_height()
            ax1.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G2_stress_by_sleep.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # --- V5 : Mean Task_Success_Rate by Sleep_Group ---
        sbs = df.groupby("Sleep_Group", observed=False)["Task_Success_Rate"].mean()

        plt.figure(figsize=(8, 5))
        ax2 = sbs.plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Task Success Rate by Sleep Group")
        plt.xlabel("Sleep Group")
        plt.ylabel("Mean Task Success Rate")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax2.patches:
            height = p.get_height()
            ax2.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G2_success_by_sleep.png"), dpi=300, bbox_inches="tight")
        plt.show()

        
        # ==== G3 – Coding_Hours_Group → Heures de code, Stress, Succès
        
        print("\n==== G3 – Group by Coding_Hours_Group → Coding, Stress & Success ====\n")

        def coding_group(hours):
            # Regroupe les heures de code en 3 catégories.
            if hours < 4:
                return "0–4h"
            elif hours <= 8:
                return "4–8h"
            else:
                return ">8h"

        df["Coding_Hours_Group"] = df["Hours_Coding"].apply(coding_group)

        g3_stats = df.groupby("Coding_Hours_Group")[[
            "Task_Success_Rate",
            "Stress_Level",
            "Errors"
        ]].mean().round(2)

        print("Mean stats by Coding_Hours_Group :")
        print(g3_stats)

        # --- Stress moyen vs heures de code ---
        plt.figure(figsize=(8, 5))
        ax1 = g3_stats["Stress_Level"].plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Stress Level by Coding Hours Group")
        plt.xlabel("Coding Hours Group")
        plt.ylabel("Mean Stress Level")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax1.patches:
            height = p.get_height()
            ax1.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G3_stress_by_coding.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # --- Taux de succès moyen vs heures de code ---
        plt.figure(figsize=(8, 5))
        ax2 = g3_stats["Task_Success_Rate"].plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Task Success Rate by Coding Hours Group")
        plt.xlabel("Coding Hours Group")
        plt.ylabel("Mean Task Success Rate")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax2.patches:
            height = p.get_height()
            ax2.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G3_success_by_coding.png"), dpi=300, bbox_inches="tight")
        plt.show()

        
        # ==== G4 – High_AI_Usage → IA, Erreurs & Succès (V6 + V7) ====
        
        print("\n==== G4 – Group by High_AI_Usage → AI, Errors & Success ====\n")

        median_ai = df["AI_Usage_Hours"].median()
        df["High_AI_Usage"] = df["AI_Usage_Hours"].apply(
            lambda x: "High_AI_Usage" if x >= median_ai else "Low_AI_Usage"
        )

        g4_stats = df.groupby("High_AI_Usage")[[
            "Errors",
            "Task_Success_Rate",
            "Stress_Level"
        ]].mean().round(2)

        print("Mean stats by High_AI_Usage :")
        print(g4_stats)

        # --- Erreurs moyennes vs IA ---
        plt.figure(figsize=(8, 5))
        ax1 = g4_stats["Errors"].plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Errors by AI Usage")
        plt.xlabel("AI Usage Group")
        plt.ylabel("Mean Errors")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax1.patches:
            height = p.get_height()
            ax1.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G4_errors_by_ai.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # --- Taux de succès moyen vs IA ---
        plt.figure(figsize=(8, 5))
        ax2 = g4_stats["Task_Success_Rate"].plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Task Success Rate by AI Usage")
        plt.xlabel("AI Usage Group")
        plt.ylabel("Mean Task Success Rate")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax2.patches:
            height = p.get_height()
            ax2.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G4_success_by_ai.png"), dpi=300, bbox_inches="tight")
        plt.show()

        
        # ==== G5 – High_Coffee → Café, Stress & Succès (bonus) ======
        
        print("\n==== G5 – Group by High_Coffee → Coffee, Stress & Success ====\n")

        med = df["Coffee_Intake"].median()
        df["High_Coffee"] = df["Coffee_Intake"].apply(
            lambda x: "High_Coffee" if x >= med else "Low_Coffee"
        )

        g5_stats = df.groupby("High_Coffee")[["Stress_Level", "Task_Success_Rate"]].mean().round(2)

        print(f"Median Coffee_Intake = {med:.2f}")
        print("\nMean stats by High_Coffee :")
        print(g5_stats)

        # Stress moyen par groupe de café
        plt.figure(figsize=(8, 5))
        ax1 = g5_stats["Stress_Level"].plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Stress Level by Coffee Intake Group")
        plt.xlabel("Coffee Group (Low vs High)")
        plt.ylabel("Mean Stress Level")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax1.patches:
            height = p.get_height()
            ax1.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G5_stress_by_coffee.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # Succès moyen par groupe de café
        plt.figure(figsize=(8, 5))
        ax2 = g5_stats["Task_Success_Rate"].plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85
        )
        plt.title("Mean Task Success Rate by Coffee Intake Group")
        plt.xlabel("Coffee Group (Low vs High)")
        plt.ylabel("Mean Task Success Rate")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax2.patches:
            height = p.get_height()
            ax2.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G5_success_by_coffee.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # ============================================================
        # ==== G6 – Sleep_Group × High_Stress → Profils combinés =====
        # ============================================================
        print("\n==== G6 – Sleep_Group × High_Stress → Sleep + Stress Profiles ====\n")

        stress_threshold = df["Stress_Level"].mean()
        df["High_Stress"] = df["Stress_Level"] > stress_threshold

        sleep_stress_summary = (
            df.groupby(["Sleep_Group", "High_Stress"])
            .agg({
                "Task_Success_Rate": "mean",
                "Errors": "mean"
            })
            .round(2)
        )

        print("Mean Task_Success_Rate & Errors by Sleep_Group × High_Stress :")
        print(sleep_stress_summary)

        pivot_success = df.pivot_table(
            values="Task_Success_Rate",
            index="High_Stress",
            columns="Sleep_Group",
            aggfunc="mean"
        ).round(2)

        print("\nPivot – Mean Task_Success_Rate by Sleep_Group & High_Stress :")
        print(pivot_success)

        fig, ax = plt.subplots(figsize=(8, 5))
        pivot_success.plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85,
            ax=ax
        )
        ax.set_title("Mean Task Success Rate by Sleep Group and Stress Level")
        ax.set_xlabel("High Stress (False = Low/Normal, True = High)")
        ax.set_ylabel("Mean Task Success Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Sleep Group")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G6_success_sleep_stress.png"), dpi=300, bbox_inches="tight")
        plt.show()

       
        # ==== G7 – High_AI_Usage × High_Stress → IA + Stress =========
        
        print("\n==== G7 – High_AI_Usage × High_Stress → AI + Stress ====\n")

        ai_threshold = df["AI_Usage_Hours"].mean()
        df["High_AI_Usage_Bool"] = df["AI_Usage_Hours"] > ai_threshold  # booléen

        ai_stress_summary = (
            df.groupby(["High_AI_Usage_Bool", "High_Stress"])
            .agg({
                "Errors": "mean",
                "Task_Success_Rate": "mean"
            })
            .round(2)
        )

        print("Mean Errors & Success by High_AI_Usage × High_Stress :")
        print(ai_stress_summary)

        pivot_errors = df.pivot_table(
            values="Errors",
            index="High_Stress",
            columns="High_AI_Usage_Bool",
            aggfunc="mean"
        ).round(2)

        print("\nPivot – Mean Errors by High_AI_Usage & High_Stress :")
        print(pivot_errors)

        fig, ax2 = plt.subplots(figsize=(8, 5))
        pivot_errors.plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85,
            ax=ax2
        )
        ax2.set_title("Mean Errors by AI Usage and Stress Level")
        ax2.set_xlabel("High Stress (False = Low/Normal, True = High)")
        ax2.set_ylabel("Mean Errors")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend(title="High AI Usage (False/True)")
        ax2.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax2.patches:
            height = p.get_height()
            ax2.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G7_errors_ai_stress.png"), dpi=300, bbox_inches="tight")
        plt.show()

        
        # ==== G8 – Coding_Hours_Group × High_Success → Profil final ==
       
        print("\n==== G8 – Coding_Hours_Group × High_Success → Coding + Success Profiles ====\n")

        success_median = df["Task_Success_Rate"].median()
        df["High_Success"] = df["Task_Success_Rate"] >= success_median

        g8_summary = df.groupby(
            ["Coding_Hours_Group", "High_Success"]
        )[["Cognitive_Load", "Bugs_Found"]].mean().round(2)

        print("Mean Cognitive_Load & Bugs_Found by Coding_Hours_Group × High_Success :")
        print(g8_summary)

        g8_pivot = g8_summary.reset_index().pivot(
            index="Coding_Hours_Group",
            columns="High_Success",
            values="Cognitive_Load"
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        g8_pivot.plot(
            kind="bar",
            edgecolor="black",
            alpha=0.85,
            ax=ax
        )
        ax.set_title("Average Cognitive Load by Coding Hours and Success")
        ax.set_xlabel("Coding Hours Group")
        ax.set_ylabel("Average Cognitive Load")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="High Success")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G8_cognitive_load_coding_success.png"), dpi=300, bbox_inches="tight")
        plt.show()


    def filtering(self):

        """
        Applique tous les filtres F1 à F7 sur le DataFrame self.df.

        F1 : High / Low Success          (Task_Success_Rate)
        F2 : High / Low Stress           (Stress_Level)
        F3 : Low / High Sleep            (Sleep_Hours)
        F4 : Heavy / Light Coders        (Hours_Coding)
        F5 : High / Low AI Usage         (AI_Usage_Hours, seuil = médiane)
        F6 : High / Low Errors           (Errors, seuil = médiane)
        F7 : High / Low Coffee           (Coffee_Intake, seuil = médiane)

        Retourne un dictionnaire contenant tous les DataFrames filtrés.
        """

        df = self.df  # alias plus court
        filters = {}

        
        # F1 – High Success / Low Success
        
        print("\n[F1] High Success / Low Success (Task_Success_Rate)")

        # High Success : Task_Success_Rate > 80
        high_success = df[df["Task_Success_Rate"] > 80]

        # Low Success : Task_Success_Rate < 60
        low_success = df[df["Task_Success_Rate"] < 60]

        print(f"  High Success  (Task_Success_Rate > 80) : {len(high_success)} Devs")
        print(f"  Low Success   (Task_Success_Rate < 60) : {len(low_success)} Devs")

        filters["high_success"] = high_success
        filters["low_success"] = low_success

       
        # F2 – High Stress / Low Stress
        
        print("\n[F2] High Stress / Low Stress (Stress_Level)")

        # High Stress : Stress_Level > 70
        high_stress = df[df["Stress_Level"] > 70]

        # Low Stress : Stress_Level < 40
        low_stress = df[df["Stress_Level"] < 40]

        print(f"  High Stress (Stress_Level > 70) : {len(high_stress)} Devs")
        print(f"  Low Stress  (Stress_Level < 40) : {len(low_stress)} Devs")

        filters["high_stress"] = high_stress
        filters["low_stress"] = low_stress

        
        # F3 – Low Sleep / High Sleep
        
        print("\n[F3] Low Sleep / High Sleep (Sleep_Hours)")

        # Low Sleep : Sleep_Hours < 6
        low_sleep = df[df["Sleep_Hours"] < 6]

        # High Sleep : Sleep_Hours > 8
        high_sleep = df[df["Sleep_Hours"] > 8]

        print(f"  Low Sleep  (Sleep_Hours < 6h) : {len(low_sleep)} Devs")
        print(f"  High Sleep (Sleep_Hours > 8h) : {len(high_sleep)} Devs")

        filters["low_sleep"] = low_sleep
        filters["high_sleep"] = high_sleep

       
        # F4 – Heavy Coders / Light Coders
       
        print("\n[F4] Heavy Coders / Light Coders (Hours_Coding)")

        # Heavy Coders : Hours_Coding > 8
        heavy_coders = df[df["Hours_Coding"] > 8]

        # Light Coders : Hours_Coding < 4
        light_coders = df[df["Hours_Coding"] < 4]

        print(f"  Heavy Coders (Hours_Coding > 8h) : {len(heavy_coders)} Devs")
        print(f"  Light Coders (Hours_Coding < 4h) : {len(light_coders)} Devs")

        filters["heavy_coders"] = heavy_coders
        filters["light_coders"] = light_coders

       
        # F5 – High AI Usage / Low AI Usage
        
        print("\n[F5] High AI Usage / Low AI Usage (AI_Usage_Hours)")

        # Seuil = médiane du temps d'utilisation de l'IA
        ai_median = df["AI_Usage_Hours"].median()
        print(f"  Median AI_Usage_Hours = {ai_median:.2f}")

        high_ai_usage = df[df["AI_Usage_Hours"] >= ai_median]
        low_ai_usage = df[df["AI_Usage_Hours"] < ai_median]

        print(f"  High_AI_Usage (>= médiane) : {len(high_ai_usage)} Devs")
        print(f"  Low_AI_Usage  (< médiane)  : {len(low_ai_usage)} Devs")

        filters["high_ai_usage"] = high_ai_usage
        filters["low_ai_usage"] = low_ai_usage

        
        # F6 – High Errors / Low Errors
        
        print("\n[F6] High Errors / Low Errors (Errors)")

        # Seuil choisi : médiane du nombre d'erreurs
        errors_median = df["Errors"].median()
        print(f"  Median Errors = {errors_median:.2f}")

        high_errors = df[df["Errors"] >= errors_median]
        low_errors = df[df["Errors"] < errors_median]

        print(f"  High Errors (>= médiane) : {len(high_errors)} Devs")
        print(f"  Low Errors  (< médiane)  : {len(low_errors)} Devs")

        filters["high_errors"] = high_errors
        filters["low_errors"] = low_errors

        
        # F7 – High Coffee / Low Coffee
        
        print("\n[F7] High Coffee / Low Coffee (Coffee_Intake)")

        # Seuil = médiane de la consommation de café
        coffee_median = df["Coffee_Intake"].median()
        print(f"  Median Coffee_Intake = {coffee_median:.2f}")

        high_coffee = df[df["Coffee_Intake"] >= coffee_median]
        low_coffee = df[df["Coffee_Intake"] < coffee_median]

        print(f"  High Coffee (>= médiane) : {len(high_coffee)} Devs")
        print(f"  Low Coffee  (< médiane)  : {len(low_coffee)} Devs")

        filters["high_coffee"] = high_coffee
        filters["low_coffee"] = low_coffee

        
        # Retour de tous les filtres
        
        return filters

    def matrix_correlation(self,afficher:bool):
        df=self.df
        df=pd.read_csv("AI_Developer_Performance_Extended_1000.csv")


        #Garde uniquement les colonnes numériques (int, float).
        numeric_cols = df.select_dtypes(include='number')


        #Calcule la matrice de corrélation entre les colonnes numériques.
        #Pearson correlation linéaire classique 
        #Spearman corrélation de rang (si les relations ne sont pas linéaires).
        corr_matrix=numeric_cols.corr(method='pearson') 


        #Arrondit les valeurs de la matrice de corrélation à 2 décimales.
        corr_rounded=corr_matrix.round(2)
        print(corr_rounded.to_string())

        #Sauvegarde la matrice de corrélation arrondie dans un fichier CSV.
        corr_rounded.to_csv("correlation_matrix.csv", index=True)



        if(afficher):
            # 1. Utiliser directement ta matrice de corrélation (déjà arrondie)
            corr_mat = corr_rounded

            # 2. Récupérer les noms des variables pour les axes
            x_labels = list(corr_mat.columns)   # noms sur l’axe X
            y_labels = list(corr_mat.index)     # noms sur l’axe Y

            # 3. Définir un style simple pour le titre (comme ton style)
            font_dict = {
                "fontsize": 14,
                "fontweight": "bold",
                "color": "purple",
                "style": "italic",
            }

            # 4. Créer la figure et l’axe (zone de tracé)
            #figsize : largeur, hauteur en pouces
            #sunbplots : crée une figure et des sous-graphes (ici 1 seul)
            fig, ax = plt.subplots(figsize=(10, 8))

            # 5. Afficher la “carte” : chaque case = une corrélation
            # vmin/vmax fixés à [-1, 1] pour une lecture correcte des corrélations
            #cmap : palette de couleurs
            #aspect="auto" : pour que les cases soient carrées même si la figure est rectangulaire
            heatmap = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

            # 6. Ajouter la légende des couleurs (colorbar)
            #colorbar : barre de couleurs à côté de la carte
            cbar = fig.colorbar(heatmap, ax=ax)
            cbar.set_label("Corrélation")

            # 7. Ajouter titre + noms des axes
            ax.set_title("Matrice de corrélation", fontdict=font_dict)
            ax.set_xlabel("Variables", fontdict=font_dict)
            ax.set_ylabel("Variables", fontdict=font_dict)

            # 8. Mettre les noms des variables sur les axes (ticks)
            ax.set_xticks(range(len(x_labels)))
            ax.set_yticks(range(len(y_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_yticklabels(y_labels)

            # 9. Ajouter une grille légère pour bien séparer les cases
            ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
            ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.tick_params(which="minor", bottom=False, left=False)

            # 10. Ajuster la mise en page pour éviter que les labels se coupent
            plt.tight_layout()
            plt.show()

        return corr_rounded
    
    def SaveCsv(self):
        # Réécrit le CSV d'origine avec la version nettoyée.
        self.df.to_csv(self.path, index=False)

    

data=Data('AI_Developer_Performance_Extended_1000.csv')
print(data)

data.inspect_data()
data.summarize_data()
data.clean_data()
data.grouping_visualization()
# filters=data.filtering()
# corr_matrix=data.matrix_correlation(True)
