import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")

# Create High_Stress column -> High_Stress = True & High_Stress = False
df['High_Stress'] = df['Stress_Level'] > 70

# Group by High_Stress & calculate the average of ....
group = df.groupby('High_Stress')[[
    'Task_Success_Rate',
    'Sleep_Hours',
    'Hours_Coding',
    'AI_Usage_Hours'
]].mean()

print(group)


#Histogramme global du stress

plt.figure(figsize=(10, 6))
plt.hist(
    df['Stress_Level'],
    bins=20,
    edgecolor='black',
    alpha=0.75
)

# Ligne verticale pour le seuil High_Stress
plt.axvline(
    70,
    color='red',
    linestyle='--',
    linewidth=2,
    label='High Stress Threshold = 70'
)

plt.title("Distribution of Stress Level with High Stress Threshold")
plt.xlabel("Stress Level")
plt.ylabel("Number of Developers")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


#Bar chart : Mean Task_Success_Rate by High_Stress

SbS = df.groupby('High_Stress')['Task_Success_Rate'].mean()

plt.figure(figsize=(7, 5))
ax = SbS.plot(
    kind='bar',
    edgecolor='black',
    alpha=0.8
)

plt.title("Mean Task Success Rate by Stress Level")
plt.xlabel("High Stress (False = Low/Medium, True = High)")
plt.ylabel("Mean Task Success Rate")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Ajouter les valeurs au-dessus des barres
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.1f}",
        (p.get_x() + p.get_width() / 2, height),
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.show()

# Explanation
'''The results show a clear difference in task success rate between stressed
 and non-stressed developers. Developers with high stress levels have a much
  lower average task success rate (38.25) compared to low or medium stress
   developers (71.76). Sleep duration, coding hours, and AI usage remain
    very similar across both groups, suggesting that stress itself plays 
    a major role in reduced performance.'''
