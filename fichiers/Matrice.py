import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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



