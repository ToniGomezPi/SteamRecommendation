import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Importamos datos del Preproceso de datos
steamGamesData = pd.read_csv("PreprocesoDatos.csv")
print(steamGamesData)

# Grafico de la distribucion por fecha, se puede ver que en el 2022 es donde hay mas juegos
plt.figure(figsize=(12, 6))
sns.countplot(data=steamGamesData, x='Release Date', order=steamGamesData['Release Date'].value_counts().index)
plt.title('Distribucion de lanzamiento del juego')  # Titulo
plt.xlabel('Año de lanzamiento')  # eje X etiqueta
plt.ylabel('Juegos')  # eje Y etiqueta
plt.xticks(rotation=90)  # Rotar etiquetas del eje X para mayor claridad
plt.show()

# Distribucion de precios de los juegos, se puede observar que de 0 a 10 esta la mayoria de juegos
bin_edges = [0, 10, 25, 50, 100,
             float('inf')]  # float('inf') actua como un predictor de cualquier numero superior a 100 en este caso
bin_labels = ['0-10', '10-25', '25-50', '50-100', '100+']  # 100+ entraran los del "float('inf')"

# Calculamos cuántos juegos hay en cada rango
price_count = pd.cut(steamGamesData['Price'], bins=bin_edges, labels=bin_labels).value_counts().sort_index()

# Grafico de barras para visualizar la distribucion de precios de los juegos
x = np.arange(len(bin_labels))

fig, ax = plt.subplots(figsize=(15, 9))

ax.set_facecolor('#2B2B2B')
fig.set_facecolor('#2B2B2B')

plt.bar(x, price_count, width=0.3, color='#31C1DE', label='Precio', edgecolor='white')

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
plt.xticks(x, bin_labels)
plt.xlabel('Rango de precios', color='white')
plt.ylabel('Juegos', color='white')
plt.title('Distribucion de juegos basados en su precio', color='white')
legend = ax.legend(facecolor='#2B2B2B', edgecolor='white')

ax.grid(axis='y', color='white', linestyle='--', linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.show()

# Numero de juegos que salen en cada año, se puede observar que coincide con nuestra primera grafica de distribucion
yearly_counts = steamGamesData['Release Date'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))

ax.set_facecolor('#2B2B2B')
fig.patch.set_facecolor('#2B2B2B')

sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, color='#FF00CE', linewidth=2, alpha=0.8, ax=ax)

ax.set_xlabel('Año de lanzamiento', fontsize=14, color='white')
ax.set_ylabel('Juegos', fontsize=14, color='white')
ax.set_title('Cantidad de juegos lanzados a lo largo de los años', fontsize=18, fontweight='bold', color='white')

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#6A6A6A')
ax.tick_params(axis='both', which='both', colors='white', labelsize=12)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Distribucion de precios segun el año de salida del juego
average_prices = steamGamesData.groupby('Release Date')['Price'].mean()

fig, ax = plt.subplots(figsize=(10, 6))

ax.set_facecolor('#2B2B2B')
fig.patch.set_facecolor('#2B2B2B')

sns.lineplot(y=average_prices.values, x=average_prices.index, color='cyan', linewidth=2, ax=ax)

ax.set_xlabel('Año de lanzamiento', fontsize=14, color='white')
ax.set_ylabel('Precio medio (€)', fontsize=14, color='white')
ax.set_title('Precio medio a lo largo de los años', fontsize=18, fontweight='bold', color='white')

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#6A6A6A')
ax.tick_params(axis='both', which='both', colors='white', labelsize=12)

plt.tight_layout()
plt.show()

# Creamos la correlation heatmap de las columnas numerales
numerical_columns = [column for column in steamGamesData.columns if
                     steamGamesData[column].dtype in ['int64', 'float64']]

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_facecolor('#2B2B2B')
fig.patch.set_facecolor('#2B2B2B')
cmap = sns.color_palette("light:b", as_cmap=True)
sns.heatmap(
    steamGamesData[numerical_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap=cmap,
    linewidths=.5,
    cbar_kws={"label": "Correlation", "orientation": "vertical"},
    square=True,
    ax=ax
)

ax.set_title('Correlation Heatmap', fontsize=18, fontweight='bold', color='white')
ax.tick_params(axis='both', which='both', colors='white', labelsize=12)

plt.show()

# Distribucion de juegos por la cantidad de reseñas
reviews_summary_counts = steamGamesData['Reviews Summary'].value_counts()
palette = sns.color_palette("mako", len(reviews_summary_counts))

threshold = 5


# con esta funcion omite valores bajos en el grafico
def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    if pct < threshold:
        return ""
    else:
        return f"{pct:.1f}%\n({absolute:d})"  # nos devuelve una cadena en formato float


fig, ax = plt.subplots(figsize=(10, 6))

ax.set_facecolor('#395970')
fig.patch.set_facecolor('#395970')

# Se crea una lista de etiquetas que esten por encima del umbral
labels = [label if (value / reviews_summary_counts.sum()) * 100 >= threshold else '' for label, value in
          zip(reviews_summary_counts.index, reviews_summary_counts)]
plt.pie(
    reviews_summary_counts,
    labels=labels,
    colors=palette,
    autopct=lambda pct: func(pct, reviews_summary_counts),
    startangle=90,
    pctdistance=0.8,
    textprops={'color': 'white'},
    wedgeprops=dict(width=0.5, edgecolor='#395970')
)

total_reviews = reviews_summary_counts.sum()

# Añadimos la leyenda con el numero de juegos y su porcentaje
legend_labels = [f'{label}: {count} ({count / total_reviews * 100:.1f}%)' for label, count in
                 zip(reviews_summary_counts.index, reviews_summary_counts)]
legend = plt.legend(legend_labels, title='Reviews Summary', loc='best', bbox_to_anchor=(1, 0.5), facecolor='#2B2B2B',
                    edgecolor='white', title_fontsize=12)
plt.setp(legend.get_texts(), color='white')
plt.setp(legend.get_title(), color='white')

plt.title('Distribucion de juegos por Reviews Summary', fontsize=18, fontweight='bold', color='white')
plt.gca().add_artist(plt.Circle((0, 0), 0.6, fc='#395970'))
plt.axis('equal')

plt.show()

# Distribucion de como califican a los juegos de los desarrolladores
developer_reviews_mode = steamGamesData.groupby('Developer')['Reviews Summary'].agg(pd.Series.mode)
print(developer_reviews_mode)

# Distribucion del numero de reseñas que hay en cada porcentaje
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_facecolor('#2B2B2B')
fig.patch.set_facecolor('#2B2B2B')
ax.grid(False)

hb = plt.hexbin(steamGamesData['Reviews_number'], steamGamesData['Reviews_percentage'], gridsize=50, cmap='Blues',
                bins='log', edgecolors='gray')
cb = plt.colorbar(hb, label='log10(count in bin)',
                  ax=ax)  # Nos despreocupamos de la cantidad de 0 reseñas que hay en muchos juegos y nos altere la visualizacion
cb.outline.set_edgecolor('white')
cb.ax.yaxis.set_tick_params(color='white')
cb.ax.set_yticklabels(cb.get_ticks(), color='white')

plt.xlabel('Numero de reseñas', fontsize=12, color='white')
plt.ylabel('Porcentaje de reseñas', fontsize=12, color='white')
plt.title('Numero de reseñas vs Porcentaje de reseñas', fontsize=14, color='white')
plt.xticks(fontsize=10, color='white')
plt.yticks(fontsize=10, color='white', rotation=45)

plt.tight_layout()

plt.show()

# Distribucion de los 10 desarrolladores con mas juegos y sus reseñas positivas/negativas
ordered_reviews = ["Mostly Negative", "Negative", "Mixed", "Mostly Positive", "Positive", "Overwhelmingly Positive",
                   "Unknown"]

top_developers = steamGamesData['Developer'].value_counts().head(10).index

filtered_data = steamGamesData[steamGamesData['Developer'].isin(top_developers)]

pivot_data = pd.crosstab(index=filtered_data['Developer'], columns=filtered_data['Reviews Summary'])

pivot_data = pivot_data[ordered_reviews]

fig, ax = plt.subplots(figsize=(15, 7))

colors = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=pivot_data.shape[1])

pivot_data.plot(kind='bar', stacked=True, ax=ax, color=colors, zorder=2)

ax.set_facecolor('#2B2B2B')
fig.patch.set_facecolor('#2B2B2B')

ax.set_title('Distribucion de reseñas sobre los Top 10 desarrolladores', fontsize=18, fontweight='bold', color='white')
ax.set_xlabel('desarrolladores', fontsize=14, color='white')
ax.set_ylabel('Juegos', fontsize=14, color='white')

leg = ax.legend(frameon=True, fontsize=12, loc='upper right', facecolor='#4A4A4A', edgecolor='white')
for text in leg.get_texts():
    text.set_color("white")

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#6A6A6A')
ax.tick_params(axis='both', which='both', colors='white', labelsize=12)

plt.tight_layout()
plt.show()
# se puede observar en la grafica que el desarrollador Dnovel tiene unas muy buenas reseñas de sus juegos

print(steamGamesData['Popular Tags'])
steamGamesData.to_csv('Analisis_datos.csv', index=False)
