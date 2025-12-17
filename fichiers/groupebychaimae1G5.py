import pandas as pd
import matplotlib.pyplot as plt

# 1) Lire le CSV
df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")

# 2) Créer High_Coffee (F7)
# ---------------------------------------
# Idée :
#   - On calcule la médiane de Coffee_Intake
#   - Si consommation >= médiane → "High_Coffee"
#   - Sinon → "Low_Coffee"
med = df["Coffee_Intake"].median()

df["High_Coffee"] = df["Coffee_Intake"].apply(
    lambda x: "High_Coffee" if x >= med else "Low_Coffee"
)

# 3) Grouping : stress moyen + succès moyen
# ---------------------------------------
# On calcule, pour chaque groupe (Low_Coffee / High_Coffee) :
#   - Stress_Level moyen
#   - Task_Success_Rate moyen
summary = df.groupby("High_Coffee")[["Stress_Level", "Task_Success_Rate"]].mean().round(2)

print("Median Coffee_Intake =", med)
print("\nTableau des moyennes (Stress & Success par groupe de café) :")
print(summary)

# 4) Bar chart 1 : Mean Stress_Level by High_Coffee
# ---------------------------------------
plt.figure(figsize=(8, 5))
ax1 = summary["Stress_Level"].plot(
    kind="bar",
    edgecolor="black",
    alpha=0.85
)

plt.title("Mean Stress Level by Coffee Intake Group")
plt.xlabel("Coffee Group (Low vs High)")
plt.ylabel("Mean Stress Level")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Afficher la valeur au-dessus de chaque barre
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
plt.show()

# 5) Bar chart 2 (optionnel) : Mean Task_Success_Rate by High_Coffee
# ---------------------------------------
plt.figure(figsize=(8, 5))
ax2 = summary["Task_Success_Rate"].plot(
    kind="bar",
    edgecolor="black",
    alpha=0.85
)

plt.title("Mean Task Success Rate by Coffee Intake Group")
plt.xlabel("Coffee Group (Low vs High)")
plt.ylabel("Mean Task Success Rate")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Afficher la valeur au-dessus de chaque barre
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
plt.show()

# Interprétation possible (à mettre dans le rapport / slides) :
"""
G5 – High_Coffee / Low_Coffee

Ce grouping compare le niveau de stress moyen et le taux de succès moyen
entre deux profils :
  • Low_Coffee : développeurs qui consomment moins de café que la médiane
  • High_Coffee : développeurs qui consomment plus de café que la médiane

Les graphiques permettent de voir si :
  - les gros buveurs de café sont plus ou moins stressés,
  - la consommation de café a vraiment un impact visible sur la performance.
"""