# Créer High_AI_Usage
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")
median_ai = df["AI_Usage_Hours"].median()

df["High_AI_Usage"] = df["AI_Usage_Hours"].apply(
    lambda x: "High_AI_Usage" if x >= median_ai else "Low_AI_Usage"
)

# Calcul des moyennes par usage IA
g4_stats = df.groupby("High_AI_Usage")[[
    "Errors",
    "Task_Success_Rate",
    "Stress_Level"
]].mean().round(2)

print(g4_stats)


# Erreurs moyennes vs IA

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

# Afficher les valeurs au-dessus des barres
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(
        f"{height:.2f}",
        (p.get_x() + p.get_width()/2, height),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()


# Succès moyen vs IA

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

# Afficher les valeurs au-dessus des barres
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(
        f"{height:.2f}",
        (p.get_x() + p.get_width()/2, height),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()

# Interprétation G4 – Utilisation de l’IA, erreurs et succès
'''The analysis shows that developers who heavily use artificial intelligence
tend to make fewer errors on average and achieve a higher task success rate
compared to those with low AI usage. The stress level is also slightly lower
among high AI users. These results suggest that AI can be an effective support
tool, improving both work quality and overall developer performance.'''
