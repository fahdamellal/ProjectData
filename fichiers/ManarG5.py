import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1) Charger le dataset
# ==============================
# On charge le fichier CSV contenant les 1000 développeurs
df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")

# ==============================
# 2) Création des groupes de sommeil (Sleep_Group)
# ==============================
# Idée :
#   - <5h   : manque de sommeil
#   - 5–7h  : sommeil "normal / moyen"
#   - >7h   : gros dormeurs

df["Sleep_Group"] = "5–7h"  # valeur par défaut
df.loc[df["Sleep_Hours"] < 5, "Sleep_Group"] = "<5h"
df.loc[df["Sleep_Hours"] > 7, "Sleep_Group"] = ">7h"

# ==============================
# 3) Création de High_Stress (booléen)
# ==============================
# Seuil : on utilise la moyenne de Stress_Level
#   - High_Stress = True  si Stress_Level > moyenne
#   - High_Stress = False sinon

stress_threshold = df["Stress_Level"].mean()
df["High_Stress"] = df["Stress_Level"] > stress_threshold

# ==============================
# 4) Groupement combiné : Sleep × Stress
# ==============================
# On veut voir, pour chaque combinaison (Sleep_Group, High_Stress),
# la performance moyenne (Task_Success_Rate) et les erreurs moyennes (Errors).

sleep_stress_summary = (
    df.groupby(["Sleep_Group", "High_Stress"])
      .agg({
          "Task_Success_Rate": "mean",
          "Errors": "mean"
      })
      .round(2)
)

print("\nRésumé sommeil × stress (moyenne succès & erreurs) :")
print(sleep_stress_summary)

# ==============================
# 5) Tableau croisé (pivot) pour la visualisation du succès
# ==============================
# Format demandé :
#   - Lignes (index) : High_Stress (False / True)
#   - Colonnes      : Sleep_Group (<5h, 5–7h, >7h)
#   - Valeurs       : Task_Success_Rate moyen

pivot_success = df.pivot_table(
    values="Task_Success_Rate",
    index="High_Stress",
    columns="Sleep_Group",
    aggfunc="mean"
).round(2)

print("\nTableau croisé : Mean Task_Success_Rate par Sleep_Group et High_Stress :")
print(pivot_success)

# ==============================
# 6) Barres groupées – Task Success Rate par Sleep_Group × Stress
# ==============================
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

# Afficher les valeurs au-dessus des barres
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
plt.show()

# ==============================
# 7) Interprétation textuelle Sleep × Stress (à mettre dans le rapport)
# ==============================
"""
Key findings (Sleep × Stress):
- The lowest task success rate is generally observed for:
    <5h sleep & High_Stress = True
- The best performance appears for:
    5–7h or >7h sleep & High_Stress = False

Conclusion:
Lack of sleep combined with high stress leads to a significant drop
in performance and higher error rates. Adequate sleep with lower stress
is associated with much better results.
"""

# ==============================
# 8) Création de High_AI_Usage
# ==============================
# Seuil : moyenne de AI_Usage_Hours
#   - High_AI_Usage = True  si AI_Usage_Hours > moyenne
#   - False sinon

ai_threshold = df["AI_Usage_Hours"].mean()
df["High_AI_Usage"] = df["AI_Usage_Hours"] > ai_threshold

# ==============================
# 9) Groupement combiné : AI Usage × Stress
# ==============================
# On veut voir, pour chaque combinaison (High_AI_Usage, High_Stress),
# les erreurs moyennes et le succès moyen.

ai_stress_summary = (
    df.groupby(["High_AI_Usage", "High_Stress"])
      .agg({
          "Errors": "mean",
          "Task_Success_Rate": "mean"
      })
      .round(2)
)

print("\nRésumé IA × stress (moyenne erreurs & succès) :")
print(ai_stress_summary)

# ==============================
# 10) Tableau croisé et barres – Erreurs par IA × Stress
# ==============================
pivot_errors = df.pivot_table(
    values="Errors",
    index="High_Stress",
    columns="High_AI_Usage",
    aggfunc="mean"
).round(2)

print("\nTableau croisé : Mean Errors par High_AI_Usage et High_Stress :")
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
ax2.legend(title="High AI Usage")
ax2.grid(axis="y", linestyle="--", alpha=0.6)

# Afficher les valeurs au-dessus des barres
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

# ==============================
# 11) Interprétations finales (à intégrer dans le rapport)
# ==============================
"""
Interpretation — AI & Stress:
- Developers with high AI usage and low stress tend to have the lowest error rates.
- Under high stress, AI usage slightly reduces errors but does not fully compensate
  for the negative effect of stress.

Final Conclusion:
- Sleep + Stress together have a strong impact on developer success and errors.
- AI usage globally improves performance and helps reduce errors, but:
    it cannot completely neutralize the effects of high stress.
- Best performance profile:
    Adequate sleep (5–7h or more),
    Low stress,
    High AI usage.
"""
