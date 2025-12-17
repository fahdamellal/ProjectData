import pandas as pd
import matplotlib.pyplot as plt

# 1) Charger les données
df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")

# Vérifier les colonnes (debug, tu peux commenter après)
print("Colonnes disponibles :", df.columns)

# 2) Créer les groupes d'usage IA (F5) et de café (F7)

# F5 : High / Low AI Usage à partir de AI_Usage_Hours
ai_median = df["AI_Usage_Hours"].median()
df["High_AI_Usage"] = df["AI_Usage_Hours"].apply(
    lambda x: "High_AI_Usage" if x >= ai_median else "Low_AI_Usage"
)

# F7 : High / Low Coffee à partir de Coffee_Intake
coffee_median = df["Coffee_Intake"].median()
df["High_Coffee"] = df["Coffee_Intake"].apply(
    lambda x: "High_Coffee" if x >= coffee_median else "Low_Coffee"
)

# Hours_Coding, Task_Success_Rate, Cognitive_Load, Bugs_Found

def coding_hours_group(hours):
    if hours < 4:
        return "0–4h"
    elif hours <= 8:
        return "4–8h"
    else:
        return ">8h"

# Groupe d'heures de code
df["Coding_Hours_Group"] = df["Hours_Coding"].apply(coding_hours_group)

# High_Success basé sur la médiane
success_median = df["Task_Success_Rate"].median()
df["High_Success"] = df["Task_Success_Rate"] >= success_median

# Moyennes par (Coding_Hours_Group, High_Success)
g8_summary = df.groupby(
    ["Coding_Hours_Group", "High_Success"]
)[["Cognitive_Load", "Bugs_Found"]].mean().round(2)

print("\nG8 – Coding Hours × Success")
print(g8_summary)

# Table pivot pour la viz (on affiche la charge cognitive)
g8_pivot = g8_summary.reset_index().pivot(
    index="Coding_Hours_Group",
    columns="High_Success",
    values="Cognitive_Load"
)

# Visualisation – UN SEUL PLOT, bien propre
fig, ax = plt.subplots(figsize=(8, 5))     # on crée la figure ET les axes

g8_pivot.plot(
    kind="bar",
    edgecolor="black",
    alpha=0.85,
    ax=ax                                   # on DIT à pandas d'utiliser CET axe
)

ax.set_title("Average Cognitive Load by Coding Hours and Success")
ax.set_xlabel("Coding Hours Group")
ax.set_ylabel("Average Cognitive Load")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="High Success")
ax.grid(axis="y", linestyle="--", alpha=0.6)

# Ajouter les valeurs au-dessus des barres
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.2f}",
        (p.get_x() + p.get_width() / 2, height),
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.show()
