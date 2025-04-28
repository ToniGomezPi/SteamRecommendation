import pandas as pd
import numpy as np
from datetime import datetime
import re

# mejor lectura de las columnas del dataframe
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)

# Leemos el dataframe
steamGamesData = pd.read_csv("merged_data.csv")
print(steamGamesData)
# Movemos los enlaces al final
link_column = steamGamesData['Link']  # Hacemos una copia
steamGamesData = steamGamesData.drop(columns=['Link'])  # Borramos
steamGamesData['Link'] = link_column  # Volvemos a meter de nuevo los enlaces y esta vez estaran al final


# Las fechas que no esten dentro del periodo del 15-8-23 o antes seran borradas junto al juego de ellas. Informacion irrelevante
# para comparar juegos y determinar cual podria gustarle al usuario final
def conversion_fecha_lanzamiento(fecha_str):
    if pd.isnull(fecha_str) or fecha_str in ['Coming Soon', 'To be announced']:
        return np.nan
    else:
        try:
            date = datetime.strptime(fecha_str, '%d %b, %Y')
            if date > datetime(2023, 8, 15):
                return np.nan
            else:
                return date.year
        except ValueError:
            try:
                return datetime.strptime(fecha_str, '%b %Y').year
            except ValueError:
                return fecha_str


steamGamesData['Release Date'] = steamGamesData['Release Date'].apply(conversion_fecha_lanzamiento)
# Borramos la columna del precio original para dejar solo el precio final con el que estan ahora
steamGamesData = steamGamesData.drop(columns=['Original Price'])
steamGamesData = steamGamesData.rename(columns={'Discounted Price': 'Price'})


# Convertimos los dolares en euros
def conversion_dolares_euros(precio_str):
    if pd.isnull(precio_str) or precio_str == 'Free':
        return '0'
    else:
        return round(float(precio_str.replace('$', '').replace(',', '')) * 0.88, 2)


steamGamesData['Price'] = steamGamesData['Price'].apply(conversion_dolares_euros)
# Fusionamos las dos columnas de review number, quitamos los NaN y borramos las columnas originales para introducir una
# nueva columna que los agrupe y en caso de que tenga recientes siempre vaya a por el más reciente
steamGamesData['Recent_or_All_Reviews'] = np.where(
    steamGamesData['Recent Reviews Number'].notna(),
    steamGamesData['Recent Reviews Number'],
    steamGamesData['All Reviews Number']
)

steamGamesData.insert(steamGamesData.columns.get_loc('All Reviews Number') + 1, 'Reviews_number',
                      steamGamesData['Recent_or_All_Reviews'])

steamGamesData.drop(columns=['Recent_or_All_Reviews'], inplace=True)

# Extraemos los porcentajes de Reviews_number y creamos la columna Reviews_percentage
steamGamesData['Reviews_percentage_delete'] = steamGamesData['Reviews_number'].str.extract(r'(\d+)%')
steamGamesData.insert(steamGamesData.columns.get_loc('Reviews_number') + 1, 'Reviews_percentage',
                      steamGamesData['Reviews_percentage_delete'])
steamGamesData.drop(columns=['Reviews_percentage_delete'], inplace=True)
steamGamesData['Extracted_Reviews'] = steamGamesData['Reviews_number'].str.extract(
    r'(\d{1,3}(?:,\d{3})*)(?= user reviews)')
# errors=coerce se usa para reemplazar cualquier valor no numerico a NaN
steamGamesData['Extracted_Reviews'] = pd.to_numeric(steamGamesData['Extracted_Reviews'].str.replace(',', ''),
                                                    errors='coerce').astype(pd.Int64Dtype())
steamGamesData['Reviews_number'] = steamGamesData['Extracted_Reviews']
steamGamesData.drop(columns=['Extracted_Reviews'], inplace=True)
steamGamesData.drop(columns=['Recent Reviews Number', 'All Reviews Number'], inplace=True)

# Limpiamos el texto de columnas
steamGamesData['Supported Languages'] = steamGamesData['Supported Languages'].str.replace(r"['\[\]]", '').str.replace(
    r"'", '')
steamGamesData['Popular Tags'] = steamGamesData['Popular Tags'].str.replace(r"['\[\]]", '').str.replace(r"'", '')
steamGamesData['Game Features'] = steamGamesData['Game Features'].str.replace(r"['\[\]]", '').str.replace(r"'", '')

# Borramos la columna de requerimientos minimos ya que no vamos a usar la información para comparar si
# puede jugar con su pc o no
steamGamesData.drop(columns=['Minimum Requirements'], inplace=True)

# Pasamos a valor numerico la fecha para reemplazar fechas NaN si pasan del 2023.
# errors='coerce' cambia todos los no numerico a NaN
steamGamesData['Release Date'] = pd.to_numeric(steamGamesData['Release Date'], errors='coerce')
current_year = 2023
steamGamesData['Release Date'] = steamGamesData['Release Date'].where(steamGamesData['Release Date'] <= current_year,
                                                                      np.nan)

# Fusionamos las dos columnas all reviews summary y recent reviews summary para tener menos valores NaN y reemplazarlos
# por una columna
steamGamesData['Reviews Summary'] = steamGamesData['Recent Reviews Summary'].fillna(
    steamGamesData['All Reviews Summary'])
steamGamesData.insert(steamGamesData.columns.get_loc('All Reviews Summary') + 1, 'Reviews Summary',
                      steamGamesData.pop('Reviews Summary'))
steamGamesData.drop(columns=['All Reviews Summary', 'Recent Reviews Summary'], inplace=True)

# El primer cuarto de las reseñas positivas es 19 nos servira para que nos elija a partir de mayor al 19% el porcentaje
# de las reseñas positivas si fuera menor seria a bastantes positivas
first_quartile_threshold = steamGamesData['Reviews_number'].quantile(0.25)


# Funcion para categorizar mejor los porcentajes de positivos y negativos de las reseñas
def atribuir_resenyas(row):
    summary = row['Reviews Summary']
    num_reviews = row['Reviews_number']

    if pd.isna(summary):
        return row

    if re.search(r'\d+ user reviews', str(summary)):
        row['Reviews Summary'] = np.nan
        row['Reviews_percentage'] = np.nan
        return row

    if pd.isna(row['Reviews_percentage']):
        if summary == 'Overwhelmingly Positive':
            row['Reviews_percentage'] = 97
        elif summary == 'Very Positive':
            row['Reviews_percentage'] = 87
        elif summary == 'Positive':
            row['Reviews_percentage'] = 89.5 if num_reviews > first_quartile_threshold else 87
        elif summary == 'Mostly Positive':
            row['Reviews_percentage'] = 74.5
        elif summary == 'Mixed':
            row['Reviews_percentage'] = 54.5
        elif summary == 'Mostly Negative':
            row['Reviews_percentage'] = 29.5
        elif summary == 'Negative':
            row['Reviews_percentage'] = 19.5 if num_reviews > first_quartile_threshold else 29.5
        elif summary == 'Very Negative':
            row['Reviews_percentage'] = 9.5
        elif summary == 'Overwhelmingly Negative':
            row['Reviews_percentage'] = 9.5
    return row


steamGamesData = steamGamesData.apply(atribuir_resenyas, axis=1)
# complete_rows = steamGamesData.dropna().shape[0] Enseña cuantas filas han sido limpiadas y procesadas

# vamos a borrar el unico titulo que tenemos NaN
# print(steamGamesData.isnull().sum()) nos enseña que no tiene nada de informacion excepto el link el cual no hay nada
print(steamGamesData[steamGamesData['Title'].isna()])
print(steamGamesData.loc[56954, 'Link'])
steamGamesData.dropna(subset=['Title'], inplace=True)

pd.set_option("future.no_silent_downcasting", True)  # Esta linea hace que el warning lo omita
steamGamesData['Reviews_number'] = steamGamesData['Reviews_number'].replace(pd.NA, np.nan)
print(steamGamesData)
# Limpiamos informacion NaN de 'Release Date'
steamGamesData = steamGamesData.dropna(subset='Release Date')

# Remplazamos valores NaN por desconocido
steamGamesData.fillna(value={'Game Description': 'Unknown', 'Developer': 'Unknown', 'Publisher': 'Unknown'},
                      inplace=True)

print(steamGamesData.isnull().sum())

# Completamos los valores NaN como Unknown y 0 para las reseñas
steamGamesData['Reviews Summary'] = steamGamesData['Reviews Summary'].fillna('Unknown')
steamGamesData['Reviews_number'] = steamGamesData['Reviews_number'].fillna(0)
steamGamesData['Reviews_percentage'] = steamGamesData['Reviews_percentage'].fillna(0)
print(steamGamesData.isnull().sum())
# Reseteamos el index para que se actualize en cada fila
steamGamesData.reset_index(drop=True, inplace=True)

print(steamGamesData)  # de 71699 juegos a 61492 limpiando datos

# Hacemos casting a int los siguientes objetos para una mayor velocidad de codigo
steamGamesData['Price'] = steamGamesData['Price'].astype(int)
steamGamesData['Reviews_number'] = steamGamesData['Reviews_number'].astype(int)
steamGamesData['Reviews_percentage'] = steamGamesData['Reviews_percentage'].astype(int)
print(steamGamesData.dtypes)

""" Al hacer un grafico de distribucion de precios segun el año veo que hay productos
que superan mas de 130€ lo cual da a sospechar y en efecto hay valores muy locos,
procedere manualmente como hay pocos voy a poner su propio precio segun steam"""
# Algunos juegos parece que por algun motivo sospechoso tienen un valor irreal los borraremos de la lista
steamGamesData.drop([35997, 36004, 36032, 36062, 36281, 36462, 59330, 59666, 59951], inplace=True)
steamGamesData.loc[13813, 'Price'] = 5
steamGamesData.loc[21298, 'Price'] = 10
steamGamesData.loc[36441, 'Price'] = 15
steamGamesData.loc[39261, 'Price'] = 10
steamGamesData.loc[39490, 'Price'] = 20
steamGamesData.loc[39685, 'Price'] = 5
steamGamesData.loc[46053, 'Price'] = 1
steamGamesData.loc[47174, 'Price'] = 1
steamGamesData.loc[51834, 'Price'] = 14
steamGamesData.loc[51835, 'Price'] = 14
steamGamesData.loc[58313, 'Price'] = 14

print(steamGamesData.query(
    "`Price`>=130"))  # otra forma de buscar en un dataframe y finalmente no hay juegos con precios desorbitantes

# Top 10 juegos por reseñas hechas al juego. Arma 3 Con el mayor numero de reseñas
print(steamGamesData[['Title', 'Reviews_number']].sort_values(by='Reviews_number', ascending=False).head(10))

# Vamos a guardar los datos ya limpios y procesados en un nuevo csv
steamGamesData.to_csv('PreprocesoDatos.csv', index=False)
