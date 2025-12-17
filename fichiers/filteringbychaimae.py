import pandas as pd

# 1) Lire le CSV
df = pd.read_csv("AI_Developer_Performance_Extended_1000.csv")

# 2) Médiane de Coffee_Intake
med = df["Coffee_Intake"].median()

# 3) Créer High_Coffee (Low si <= médiane, High si > médiane)
df["High_Coffee"] = df["Coffee_Intake"].apply(
    lambda x: "High_Coffee" if x > med else "Low_Coffee"
)

# 4) Sauvegarder
df.to_csv("data_with_high_coffee.csv", index=False)

print("Median Coffee_Intake =", med)
print("Saved: data_with_high_coffee.csv")