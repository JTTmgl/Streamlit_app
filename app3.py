import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from sklearn.preprocessing import MinMaxScaler
import random
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

# Ajoutez le titre
st.title("Analyse de la délinquance à Nogent sur Marne")


@st.cache
def charger_donnees(file_path):
    data = pd.read_csv(file_path, sep=';', decimal=',')
    return data

# Ajouter les informations de l'étudiant dans la sidebar
st.sidebar.title('Informations de l\'étudiant')
st.sidebar.write('Prénom: Jean')
st.sidebar.write('Nom: Gambart de Lignières')
st.sidebar.write('Classe: BIA1')
st.sidebar.write('#datavz2023efrei')


# Assurez-vous d'avoir importé votre fichier CSV avec les données
file_path = '/Users/jeangambart/Desktop/Mes fichiers/Cours/M1/Data Visualization/Projet/donnee-dep-data.gouv-2022-geographie2023-produit-le2023-07-17.csv'
data = pd.read_csv(file_path, sep=';', decimal=',')

# Sélectionnez 10 régions spécifiques
selected_regions = random.sample(data['Code.région'].unique().tolist(), 10)
subset_data = data[data['Code.région'].isin(selected_regions)]
scaler = MinMaxScaler()
subset_data['POP'] = scaler.fit_transform(subset_data['POP'].values.reshape(-1, 1))
subset_data['tauxpourmille'] = scaler.fit_transform(subset_data['tauxpourmille'].values.reshape(-1, 1))



# Graphique 1 : Distribution des taux de délinquance (Bokeh)
st.subheader('Distribution des taux de délinquance (Bokeh)')
p = figure(width=600, height=400, title="Distribution des taux de délinquance")
hist, edges = np.histogram(data['tauxpourmille'], bins=20)
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="skyblue", line_color="black")
p.xaxis.axis_label = "Taux de délinquance pour mille habitants"
p.yaxis.axis_label = "Nombre d'observations"
st.bokeh_chart(p)



# Graphique 2 : Répartition des classes de délits par région au fil des années (Seaborn)
st.subheader('Répartition des classes de délits par région au fil des années (Seaborn)')
reg_data = data.groupby(['Code.région', 'annee'])['tauxpourmille'].sum().unstack()

plt.figure(figsize=(12, 6))
sns.heatmap(reg_data, cmap="YlGnBu", annot=True, fmt=".1f", cbar_kws={'label': 'Taux de délinquance pour mille habitants'})
plt.title("Délinquance par région au fil des années")
plt.xlabel("Année")
plt.ylabel("Région")
st.pyplot()




# Graphique 3 : Corrélation entre les années et les taux de délinquance (Altair)
st.subheader('Corrélation entre les années et les taux de délinquance (Altair)')
alt_chart = alt.Chart(data).mark_circle(size=60, opacity=0.5).encode(
    x=alt.X('annee:O', title='Année'),
    y=alt.Y('tauxpourmille:Q', title='Taux de délinquance pour mille habitants')
).properties(
    width=600,
    height=400,
    title="Corrélation entre les années et les taux de délinquance"
).interactive()

st.altair_chart(alt_chart)



#Graphique 4 : Taux de délinquance par année et classe de délit

# Sélectionnez les classes à afficher
classes_to_plot = ["Coups et blessures volontaires", "Usage de stupéfiants", "Vols avec armes"]

# Créez un graphique en ligne pour chaque classe de délit avec des couleurs distinctes
st.subheader("Taux de délinquance par année et classe de délit (Seaborn)")

plt.figure(figsize=(14, 8))
color_palette = sns.color_palette("husl", len(classes_to_plot))  # Palette de couleurs
for i, crime_class in enumerate(classes_to_plot):
    class_data = data[data['classe'] == crime_class]
    class_data_grouped = class_data.groupby('annee')['tauxpourmille'].mean()
    sns.lineplot(x=class_data_grouped.index, y=class_data_grouped.values, label=crime_class, palette=[color_palette[i]])

plt.title("Taux de délinquance pour mille habitants par année")
plt.xlabel("Année")
plt.ylabel("Taux de délinquance pour mille habitants")
plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Affichez le graphique Seaborn
st.pyplot()



#Graphique 5 : Corrélation entre les classes de délits et les taux de délinquance 
# Sélectionnez les classes de délits uniques
classes_de_delits_uniques = data['classe'].unique()

# Créez un dictionnaire pour stocker les corrélations par classe de délit
correlations_par_classe = {}

# Calculer la corrélation pour chaque classe de délit
for classe_de_delit in classes_de_delits_uniques:
    subset_data = data[data['classe'] == classe_de_delit]
    correlation = subset_data['tauxpourmille'].corr(subset_data['faits'])
    correlations_par_classe[classe_de_delit] = correlation

# Créez un graphique en barres pour visualiser les corrélations
st.subheader("Corrélation entre les classes de délits et les taux de délinquance (Seaborn)")

plt.figure(figsize=(12, 6))
sns.barplot(x=list(correlations_par_classe.keys()), y=list(correlations_par_classe.values()), palette="YlGnBu")
plt.title("Corrélation entre les classes de délits et les taux de délinquance")
plt.xlabel("Classe de délit")
plt.ylabel("Corrélation (Pearson)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Affichez le graphique Seaborn
st.pyplot()



#Graphique 6 : Corrélation entre les taux de délinquance par département
# Convertissez la colonne 'tauxpourmille' en chaîne de caractères
data['tauxpourmille'] = data['tauxpourmille'].astype(str)

# Remplacez les virgules par des points
data['tauxpourmille'] = data['tauxpourmille'].str.replace(',', '.').astype(float)

# Divisez par 10 pour obtenir le taux pour mille
data['tauxpourmille'] = data['tauxpourmille'] / 10

# Créez un widget de sélection pour les départements
selected_departements = st.multiselect("Sélectionnez les départements à inclure dans la corrélation", data['Code.département'].unique())

# Filtrez les données en fonction de la sélection
filtered_data = data[data['Code.département'].isin(selected_departements)]

# Agrégez les données en utilisant la somme des taux de délinquance pour les mêmes années et départements
filtered_data = filtered_data.groupby(['annee', 'Code.département'])['tauxpourmille'].sum().reset_index()

# Pivoter les données pour obtenir un format approprié
filtered_data = filtered_data.pivot(index='annee', columns='Code.département', values='tauxpourmille')

# Remplacer les valeurs manquantes par 0
filtered_data = filtered_data.fillna(0)

# Calculez la matrice de corrélation entre les taux de délinquance des départements sélectionnés
correlation_matrix = np.corrcoef(filtered_data, rowvar=False)

# Créez un graphique interactif de la corrélation sous forme de carte de chaleur
st.title("Corrélation entre les taux de délinquance par département")
fig_corr = px.imshow(correlation_matrix, x=selected_departements, y=selected_departements)

# Ajoutez des valeurs sur la matrice de corrélation
fig_corr.update_traces(text=np.around(correlation_matrix, decimals=2), showscale=False)

# Personnalisez les couleurs
fig_corr.update_layout(
    coloraxis_showscale=True,
    coloraxis_colorbar_title="Corrélation",
    coloraxis_cmin=-1,  # Valeur minimale de corrélation
    coloraxis_cmax=1,   # Valeur maximale de corrélation
)

st.plotly_chart(fig_corr)