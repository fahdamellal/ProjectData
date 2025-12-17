#importer et charger les données
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")

# Vérification rapide
print(df.head())

# Créer Coding_Hours_Group
def coding_group(hours):
    if hours < 4:
        return "0-4h"
    elif hours <= 8:
        return "4-8h"
    else:
        return ">8h"

df["Coding_Hours_Group"] = df["Hours_Coding"].apply(coding_group)

# Calcul des moyennes par groupe
g3_stats = df.groupby("Coding_Hours_Group")[[
    "Task_Success_Rate",
    "Stress_Level",
    "Errors"
]].mean().round(2)

print(g3_stats)

# ==========================
# Graphe 1 : Stress moyen vs heures de code
# ==========================
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

# Afficher les valeurs au-dessus des barres
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(
        f"{height:.1f}",
        (p.get_x() + p.get_width()/2, height),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()


# Bar chart : Taux de succès moyen vs heures de code

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

# Afficher les valeurs au-dessus des barres
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(
        f"{height:.1f}",
        (p.get_x() + p.get_width()/2, height),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()

# Interprétation G3 – Heures de code, stress et succès
'''The results show that developers who code between 4 and 8 hours per day
achieve the highest average task success rate. When coding time exceeds 8 hours,
the stress level increases without a significant improvement in performance.
This suggests that a moderate workload leads to better results while keeping
stress and errors under control.'''
