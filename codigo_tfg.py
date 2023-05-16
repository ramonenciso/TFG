# Tabrajo de Fin de Grado
# Nombre: Ramón Enciso
# Curso: 4ºA Business Analytics
# Título: Tendencias dentro de los videojuegos más vemdidos de la historia

# Librerias

import pandas as pd
import numpy as np
import matplotlib as mpl
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt

# TRATAMIENTO DE LOS DATOS
# Cargar datos

vg_df = pd.read_csv("/Users/ramonencisoortega/Desktop/uni/CUARTO/TFG/CODIGO/DATA/Video_Games_Sales_as_at_22_Dec_2016.csv")
pib_mundial = pd.read_excel("/Users/ramonencisoortega/Desktop/uni/CUARTO/TFG/pib_mundial.xlsx", engine='openpyxl')
paises = pd.read_excel("/Users/ramonencisoortega/Desktop/uni/CUARTO/TFG/Países-y-Capitales-del-Mundo (1).xlsx", engine="openpyxl")
critic_score = pd.read_csv("/Users/ramonencisoortega/Desktop/uni/CUARTO/TFG/Video Games Sales.csv")
# TRANSFORMAR DATOS VIDEOJUEGOS

# Elminiar datos que no nos interesan
vg_df = vg_df.loc[vg_df["Global_Sales"] >= 1]

# Eliminar columnas que no nos interesan
vg_df = vg_df.drop(columns=["Critic_Score", "Critic_Count", "User_Score", "User_Count", "Developer", "Rating"])

# Gestionar los nulos
vg_df.isnull().sum() # Name, Year_of_release and Publisher, have some empty values.

vg_df[(vg_df["Name"].isnull())] #index 659, we delete it

vg_df = vg_df.drop(659)

vg_df[(vg_df["Year_of_Release"].isnull())] # Look on the internet the releases and add them

vg_df.loc[vg_df.index == 183, 'Year_of_Release'] = 2003
vg_df.loc[vg_df.index == 377, 'Year_of_Release'] = 2003
vg_df.loc[vg_df.index == 456, 'Year_of_Release'] = 2008
vg_df.loc[vg_df.index == 475, 'Year_of_Release'] = 2005
vg_df.loc[vg_df.index == 609, 'Year_of_Release'] = 1978
vg_df.loc[vg_df.index == 627, 'Year_of_Release'] = 2007
vg_df.loc[vg_df.index == 657, 'Year_of_Release'] = 2001
vg_df.loc[vg_df.index == 678, 'Year_of_Release'] = 2008
vg_df.loc[vg_df.index == 719, 'Year_of_Release'] = 2006
vg_df.loc[vg_df.index == 805, 'Year_of_Release'] = 2010
vg_df.loc[vg_df.index == 1131, 'Year_of_Release'] = 2010
vg_df.loc[vg_df.index == 1142, 'Year_of_Release'] = 2007
vg_df.loc[vg_df.index == 1301, 'Year_of_Release'] = 1998
vg_df.loc[vg_df.index == 1506, 'Year_of_Release'] = 1980
vg_df.loc[vg_df.index == 1538, 'Year_of_Release'] = 2008
vg_df.loc[vg_df.index == 1585, 'Year_of_Release'] = 1977
vg_df.loc[vg_df.index == 1609, 'Year_of_Release'] = 2011
vg_df.loc[vg_df.index == 1650, 'Year_of_Release'] = 2002
vg_df.loc[vg_df.index == 1699, 'Year_of_Release'] = 2002
vg_df.loc[vg_df.index == 1840, 'Year_of_Release'] = 2007
vg_df.loc[vg_df.index == 1984, 'Year_of_Release'] = 1999
vg_df.loc[vg_df.index == 2010, 'Year_of_Release'] = 1997


vg_df[(vg_df["Publisher"].isnull())] # we search on the internet the publishers

vg_df.loc[vg_df.index == 475, 'Publisher'] = 'THQ'
vg_df.loc[vg_df.index == 1301, 'Publisher'] = 'Electronic Arts'
vg_df.loc[vg_df.index == 1667, 'Publisher'] = 'GBA'

minimum = vg_df["Year_of_Release"].min()
print(minimum)

# TRANSFORMAR DATOS PIB

pib_mundial = pib_mundial.drop(columns=["Country Code", "Indicator Name", "Indicator Code", 1960])
# No quito los nulos, porque no quiero valores individuales, voy a hacer la media por continentes y tal
nuevo_nombre_pib = {"Country Name": "country name"}
pib_mundial = pib_mundial.rename(columns=nuevo_nombre_pib)
pib_mundial = pib_mundial.applymap(lambda x: x.lower() if isinstance(x, str) else x) # todo a minusculas
pib_mundial = pib_mundial.applymap(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() if isinstance(x, str) else x)
pib_mundial = pib_mundial.iloc[[170, 73, 119, 263]]
pib_mundial_final = pib_mundial.T
pib_mundial_final = pib_mundial_final.drop(index="country name")
pib_mundial_final = pib_mundial_final.rename(columns={170: "NA_PIB", 73: "EU_PIB", 119: "JP_PIB", 263: "GB_PIB"})
pib_mundial_final = pib_mundial_final.drop(index=[1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976])
ano = list(range(1977, 2022))
pib_mundial_final = pib_mundial_final.assign(Year_of_Release=ano)

# Restablecer el índice a una secuencia numérica predeterminada (que comienza desde 0)
pib_mundial_final = pib_mundial_final.reset_index(drop=True)

# Añadir 1 a todos los valores de índice para que comiencen desde 1 en lugar de 0
pib_mundial_final.index += 1

vg_df_final = pd.merge(vg_df, pib_mundial_final, on='Year_of_Release')

critic_score = critic_score[['Game Title', 'Review']]
critic_score = critic_score.rename(columns={'Game Title': 'Name'})

vg_df_final = pd.merge(vg_df_final, critic_score, on='Name')

vg_df_final = vg_df_final.groupby('Name').agg({'Platform': 'first', 'Year_of_Release': 'first', 'Genre': 'first', 'Publisher': 'first', 'NA_Sales': 'first', 'EU_Sales': 'first', 'JP_Sales': 'first', 'Other_Sales': 'first', 'Global_Sales': 'first', 'NA_PIB': 'first', 'EU_PIB': 'first', 'JP_PIB': 'first', 'GB_PIB': 'first', 'Review': 'mean'}).reset_index()
vg_df_final = vg_df_final.sort_values(by='Global_Sales', ascending=False)
vg_df_final = vg_df_final.reset_index(drop=True)
vg_df_final.index += 1

vg_df_final["GB_PIB"] = vg_df_final["GB_PIB"].astype(float)

# Paso 1: Calcular los cuartiles
def eval_category(gs):
    if gs > 10:
        return 1
    elif gs > 2:
        return 0.5
    else:
        return 0

# Paso 3: Aplicar la función a la columna Global_Sales y guardar los resultados en la nueva columna evaluation
vg_df_final['evaluation'] = vg_df_final['Global_Sales'].apply(eval_category)

########################################################################################################################

# MODELOS DE LOS DATOS

# PCA

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# KMeans
from sklearn.cluster import KMeans
import plotly.express as px


# resumir las plataformas en tipo de consola, fabricante y generación
# crear handheld, console y PC
handhelds = ['PSP', 'PSV', 'GB', 'GBA', 'DS', '3DS']
vg_df_final['Platform_Type'] = vg_df_final['Platform'].map(
    lambda x: 'Portable' if x in handhelds else 'Domestica' if x != 'PC' else 'PC'
)

# crear fabricantes
nintendo = ['Wii', 'NES', 'SNES', 'N64', 'GC', 'WiiU', 'GB', 'GBA', 'DS', '3DS', 'GBA', 'GB']
sony = ['PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV']
sega = ['DC', 'GEN', 'SCD', 'SAT']
microsoft = ['XB', 'X360', 'XOne']
atari = ['2600', 'SAT', 'GEN']
vg_df_final['Platform_Manufacturer'] = vg_df_final['Platform'].map(
    lambda x: 'Nintendo' if x in nintendo else 'Sony' if x in sony else 'Sega' if x in sega else 'Microsoft' if x in microsoft else 'Atari' if x in atari else 'PC'
)

# crear generaciones
generations = {
    'NES': 1,
    'SNES': 2,
    'N64': 3,
    'GC': 4,
    'Wii': 5,
    'WiiU': 6,
    'GB': 1,
    'GBA': 2,
    'DS': 3,
    '3DS': 4,
    'PS': 1,
    'PS2': 2,
    'PS3': 3,
    'PS4': 4,
    'PSP': 2,
    'PSV': 3,
    'XB': 1,
    'X360': 2,
    'XOne': 3,
    'PC': 1,
    '2600': 1,
    'DC': 1,
    'GEN': 1,
    'SCD': 1,
    'SAT': 1
}
vg_df_final['Generation'] = vg_df_final['Platform'].map(generations)

# eliminar columna Platform y Year_of_Release
vg_df_final.drop(['Platform', 'Year_of_Release'], axis=1, inplace=True)

# resumir los publishers en grandes compañías e independientes y en autopublicados o no
# crear lista de grandes compañías
big_companies = ['Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment', 'Ubisoft', 'Take-Two Interactive', 'THQ', 'Konami Digital Entertainment', 'Sega', 'Namco Bandai Games', 'Microsoft Game Studios', 'Capcom', 'Atari', 'Square Enix', 'Warner Bros. Interactive Entertainment', 'Disney Interactive Studios', 'Eidos Interactive', 'LucasArts', 'Bethesda Softworks', 'Midway Games', 'Acclaim Entertainment', '505 Games', 'Vivendi Games', 'SquareSoft', 'GT Interactive', 'Enix Corporation', 'Virgin Interactive', 'Deep Silver', 'GT Interactive', 'Hasbro Interactive', 'NCSoft', 'RedOctane', 'Infogrames']
vg_df_final['Publisher_Type'] = vg_df_final['Publisher'].map(
    lambda x: 'Big Company' if x in big_companies else 'Independent'
)

# crear Autopublished para relaciones entre compañías y publishers usando Platform_Manufacturer
# crear diccionario de compañías y publishers
companies_publishers = {
    'Nintendo': 'Nintendo',
    'Sony Computer Entertainment': 'Sony',
    'Sony Computer Entertainment Europe': 'Sony',
    'Sony Oznline Entertainment': 'Sony',
    'Microsoft Game Studios': 'Microsoft',
    'Sega': 'Sega',
    'Atari': 'Atari'
    }
vg_df_final['Autopublished'] = vg_df_final['Publisher'].map(companies_publishers)
# si Autopublished coincide con Platform_Manufacturer, entonces es autopublicado y se cambia a 'Yes'
vg_df_final['Autopublished'] = vg_df_final.apply(lambda x: 'Yes' if x['Autopublished'] == x['Platform_Manufacturer'] else 'No', axis=1)
# se llenan los valores nulos con 'No'
vg_df_final['Autopublished'] = vg_df_final['Autopublished'].fillna('No')

# eliminar Publisher
vg_df_final.drop('Publisher', axis=1, inplace=True)

vg_df_final['Autopublished'].value_counts()

### obtener componentes principales
# one hot encoding
df_dummies = pd.get_dummies(vg_df_final, columns=['Genre', 'Platform_Type', 'Platform_Manufacturer', 'Publisher_Type', 'Autopublished'], drop_first=True)

df_pca = df_dummies.drop(['Name'], axis=1)

# estandarizar
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca)

# PCA
pca = PCA()
df_pca = pca.fit_transform(df_scaled)

# bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')
plt.show()

# varianza acumulada plot, marcar líneas horizontales en 0.8, 0.9 y 0.95
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.hlines(y=0.8, xmin=0, xmax=len(pca.explained_variance_ratio_), colors='r', linestyles='dashed')
plt.hlines(y=0.9, xmin=0, xmax=len(pca.explained_variance_ratio_), colors='g', linestyles='dashed')
plt.hlines(y=0.95, xmin=0, xmax=len(pca.explained_variance_ratio_), colors='b', linestyles='dashed')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Cumulative Explained Variance (0.8 at {}, 0.9 at {}, 0.95 at {})'.format(
    np.where(np.cumsum(pca.explained_variance_ratio_) > 0.8)[0][0],
    np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0],
    np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]
))
plt.show()

print(vg_df_final['Genre'].unique())

# KMEANS

# seleccionamos los primeros 22 componentes principales
df_pca = df_pca[:, :22]

# método del codo para selección del número de clusters
inertias = []
for num_clusters in list(range(1, 11)):
    kmeans = KMeans(n_clusters=num_clusters, n_init=30, random_state=10)
    kmeans.fit(df_pca)
    inertias.append(kmeans.inertia_)
plt.plot(inertias)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# prácticamente no hay codo, seleccionamos 6 clusters que es donde apenas se aprecia
kmeans = KMeans(n_clusters=6, n_init=30, random_state=10)
kmeans.fit(df_pca)
clusters = kmeans.predict(df_pca)

# añadir columna de clusters al dataframe
vg_df_final['Cluster'] = clusters

# tamaños de los clusters
vg_df_final['Cluster'].value_counts()

