import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")

# Create Sleep_Group column with 3 categories <5h, 5–7h, >7h
df['Sleep_Group'] = pd.cut(
    df['Sleep_Hours'],
    bins=[0, 5, 7, 24],
    labels=['<5h', '5–7h', '>7h']
)

# Group by Sleep_Group & calculate the average of ....
sleep_group = df.groupby('Sleep_Group', observed=False)[[
    'Stress_Level',
    'Task_Success_Rate',
    'Errors'
]].mean()

print(sleep_group)


# Histogramme du taux de succès

plt.figure(figsize=(10, 6))
plt.hist(
    df['Task_Success_Rate'],
    bins=20,
    edgecolor='black',
    alpha=0.8
)
plt.title("Distribution of Task Success Rate")
plt.xlabel("Task Success Rate")
plt.ylabel("Number of Developers")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Bar chart : Mean Stress_Level by Sleep_Group

ss = df.groupby('Sleep_Group', observed=False)['Stress_Level'].mean()

plt.figure(figsize=(8, 5))
ax1 = ss.plot(
    kind='bar',
    edgecolor='black',
    alpha=0.85
)
plt.title("Mean Stress Level by Sleep Group")
plt.xlabel("Sleep Group")
plt.ylabel("Mean Stress Level")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Ajouter les valeurs au-dessus des barres
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(
        f"{height:.1f}",
        (p.get_x() + p.get_width() / 2, height),
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.show()

# Bar chart : Mean Task_Success_Rate by Sleep_Group

sbs = df.groupby('Sleep_Group', observed=False)['Task_Success_Rate'].mean()

plt.figure(figsize=(8, 5))
ax2 = sbs.plot(
    kind='bar',
    edgecolor='black',
    alpha=0.85
)
plt.title("Mean Task Success Rate by Sleep Group")
plt.xlabel("Sleep Group")
plt.ylabel("Mean Task Success Rate")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Ajouter les valeurs au-dessus des barres
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(
        f"{height:.1f}",
        (p.get_x() + p.get_width() / 2, height),
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.show()

# Explanation
'''The results show that developers who sleep less than 5 hours have higher 
stress levels and a lower average task success rate compared to other groups.
 Developers sleeping between 5 and 7 hours show moderate stress and improved 
 performance. The group sleeping more than 7 hours appears to be the most 
 balanced, with lower stress and higher task success.
  This suggests that adequate sleep has a positive impact on both well-being 
  and performance.'''
