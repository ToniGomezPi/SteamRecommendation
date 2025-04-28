import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import csv
import re
import ast
nltk.download('stopwords')

# mejor lectura de las columnas del dataframe
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)

# Importamos datos de los graficos
steamGamesData = pd.read_csv('Analisis_datos.csv')
#print(steamGamesData)

# Tarda 7~8 minutos para que procese los keywords

# Palabras que omitimos ( como a, an, the... )
stop = set(stopwords.words('english'))
# Filtro para quitar signos de puntuacion ( como ; : ! , )
exclude = set(string.punctuation)

# Quitamos los signos de puntuacion y palabras redundantes
def clean(doc):
    tokens = gensim.utils.simple_preprocess(doc, deacc=True, min_len=3)
    stop_free = " ".join([i for i in tokens if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    return punc_free.split()

# Reemplazamos la descripcion de los juegos
steamGamesData['cleaned_description'] = steamGamesData['Game Description'].fillna('').apply(clean)

dictionary = corpora.Dictionary(steamGamesData['cleaned_description'])
# Convierte la descripcion a BoW formato, una representacion numerica para que ldamodel entienda la semantica
corpus = [dictionary.doc2bow(text) for text in steamGamesData['cleaned_description']]

# Modelo de entrenamiento que coge directamente palabras importantes de cada descripcion dada
ldamodel = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)

# Funcion que asigna temas dominantes de las descripciones

def get_dominant_topic(ldamodel, corpus, texts):
    dominant_topic = []

    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        dominant_topic.append(int(row[0][0]))

    return dominant_topic


steamGamesData['Topics_from_description'] = get_dominant_topic(ldamodel, corpus, steamGamesData['cleaned_description'])

# Nos devuelve las palabras clave de los temas mas dominantes de cada descripción
def get_dominant_topic_keywords(ldamodel, corpus, num_keywords=5):
    dominant_topic_keywords = []

    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        topic_num = int(row[0][0])
        topic_words = [word for word, prob in ldamodel.show_topic(topic_num, topn=num_keywords)]

        dominant_topic_keywords.append(", ".join(topic_words))

    return dominant_topic_keywords

# incorporamos las palabras claves a una nueva columna
steamGamesData['Topics_from_description'] = get_dominant_topic_keywords(ldamodel, corpus)

topics = steamGamesData['Topics_from_description'].value_counts()

print(steamGamesData['Topics_from_description'])

plt.figure(figsize=(10,6))
topics.plot(kind='bar', color='skyblue')
plt.title('Distribucion de palabras clave en la descripción del juego')
plt.xlabel('Palabras clave')
plt.ylabel('Juegos')
plt.show()

print(steamGamesData)
print(steamGamesData['Popular Tags'].unique())

# Mapeamos la calidad de las reseñas
review_mapping = {
    'Overwhelmingly Positive': 9,
    'Very Positive': 8,
    'Positive': 7,
    'Mostly Positive': 6,
    'Mixed': 5,
    'Mostly Negative': 4,
    'Negative': 3,
    'Very Negative': 2,
    'Overwhelmingly Negative': 1,
    'Unknown': 0
}

steamGamesData['Reviews_Summary_Numeric'] = steamGamesData['Reviews Summary'].map(review_mapping)

plt.figure(figsize=(10,6))
steamGamesData['Reviews_Summary_Numeric'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
plt.title('Distribucion de reseñas')
plt.xlabel('Categoria de reseñas')
plt.ylabel('Juegos')
plt.show()

print(steamGamesData.shape)
for column in steamGamesData.columns:
    print(column)


# Agregamos la cantidad de reviews por desarrollador por la mediana
train_data, test_data = train_test_split(steamGamesData, test_size=0.2, random_state=42)

dev_means = train_data.explode('Developer')['Developer'].reset_index().merge(
    train_data[['Reviews_number']],
    left_on='index',
    right_index=True
).groupby('Developer')['Reviews_number'].mean().to_dict()

global_mean = train_data['Reviews_number'].mean()

# Calcula la mediana de reviews_number para cada desarrollador
def target_encode_devs(row):
    devs = str(row['Developer']).split(", ")
    return np.mean([dev_means.get(dev, global_mean) for dev in devs])

# aplicamos la funcion para calcular la mediana de reviews_number que luego entrenaremos
train_data['Developer_encoded'] = train_data.apply(target_encode_devs, axis=1)
test_data['Developer_encoded'] = test_data.apply(target_encode_devs, axis=1)

# si el developer esta en dev_means se le da la mediana si no se le aplica global_mean
steamGamesData['Developer_encoded'] = steamGamesData['Developer'].apply(lambda x: np.mean([dev_means.get(dev, global_mean) for dev in str(x).split(", ")]))

print(steamGamesData)

# Hacemos lo mismo que con developer, calculamos la media para los publishers
train_data, test_data = train_test_split(steamGamesData, test_size=0.2, random_state=42)

pub_means = train_data.groupby('Publisher')['Reviews_number'].mean().to_dict()

steamGamesData['Publisher_encoded_target'] = steamGamesData['Publisher'].map(pub_means).fillna(0)

train_data['Publisher_encoded_target'] = train_data['Publisher'].map(pub_means).fillna(0)
test_data['Publisher_encoded_target'] = test_data['Publisher'].map(pub_means).fillna(0)

print(steamGamesData)

print(steamGamesData.isnull().sum())
steamGamesData['Popular Tags'] = steamGamesData['Popular Tags'].apply(lambda x: x[1:-1].split(', '))
steamGamesData['Popular Tags'] = steamGamesData['Popular Tags'].apply(lambda x: ' '.join(x))


vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(steamGamesData['Popular Tags'])


n_components = 200
svd = TruncatedSVD(n_components=n_components)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# total de cantidad de variancia(dispersion)
print("Cumulative explained variance:", svd.explained_variance_ratio_.cumsum()[-1])
tfidf_df = pd.DataFrame(tfidf_reduced, columns=[f'tag_pc_{i}' for i in range(tfidf_reduced.shape[1])])
steamGamesData = pd.concat([steamGamesData, tfidf_df], axis=1)

steamGamesData.drop(columns=['Popular Tags'], inplace=True)
print(steamGamesData)

print(steamGamesData['Supported Languages'].unique())

steamGamesData['Supported Languages'] = steamGamesData['Supported Languages'].apply(lambda x: x[1:-1].split(', '))


print(steamGamesData['Supported Languages'][0])
print(steamGamesData['Supported Languages'])
steamGamesData['languages_str'] = steamGamesData['Supported Languages'].apply(' '.join)

tfidf = TfidfVectorizer()
languages_matrix = tfidf.fit_transform(steamGamesData['languages_str'])


n_components = 70
svd = TruncatedSVD(n_components=n_components)
languages_svd = svd.fit_transform(languages_matrix)

print("Cumulative explained variance:", svd.explained_variance_ratio_.cumsum()[-1])

for i in range(n_components):
    steamGamesData[f'language_component_{i}'] = languages_svd[:, i]

# Borramos columnas que ya no nos hacen falta
steamGamesData = steamGamesData.drop(columns=['languages_str', 'Supported Languages'])
print(steamGamesData)

steamGamesData['Game Features'] = steamGamesData['Game Features'].apply(lambda x: x[1:-1].split(', '))


def join_list(features): # diferente forma de hacerlo
    return ' '.join(features)
# juntamos la lista a un string para procesarlo en la vectorizacion
steamGamesData['Game Features Str'] = steamGamesData['Game Features'].apply(join_list)

vectorizer = TfidfVectorizer(max_df=0.95, max_features=5000, stop_words='english')
X = vectorizer.fit_transform(steamGamesData['Game Features Str'])


n_components = 50
svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(X)

print(f'Cumulative explained variance: {svd.explained_variance_ratio_.sum()}')
for i in range(n_components):
    steamGamesData[f'Game_Features_Component_{i}'] = X_reduced[:, i]

print(steamGamesData)

columns_to_remove = ['Game Description', 'Reviews Summary', 'Developer', 'Publisher',
                     'Game Features', 'cleaned_description',
                     'Topics_from_description', 'Game Features Str','Link']
# Borramos las columnas con las que ya tenemos todos los datos procesados, entranados y vectorizados
steamGamesData = steamGamesData.drop(columns=columns_to_remove)
for i in steamGamesData.columns:
    print(i)

nan_columns = steamGamesData.columns[steamGamesData.isnull().any()].tolist()
nan_count = steamGamesData[nan_columns].isnull().sum()

print(nan_count)
print(steamGamesData)



original_data = steamGamesData.copy()

titles = steamGamesData['Title']
numerical_data = steamGamesData.drop('Title', axis=1)

# Modelo de entreno que tendra una media de 0 a 1 de desviacion estandarizando los datos del dataframe.
# conseguimos datos mas consistentes y mas faciles para operar ( son numeros )
scaler = StandardScaler()

print(numerical_data)
scaled_data = scaler.fit_transform(numerical_data)

scaled_data_df = pd.DataFrame(scaled_data, columns=numerical_data.columns, index=steamGamesData.index)
scaled_data_df['Title'] = titles

steamGamesData = scaled_data_df



sample_data = steamGamesData.sample(1000, random_state=42)

cosine_sim_sample_data = sample_data.drop('Title', axis=1)

num_sample_games = cosine_sim_sample_data.shape[0]
similarity_matrix_sample = pd.DataFrame(np.zeros((num_sample_games, num_sample_games)), columns=sample_data['Title'], index=sample_data['Title'])

for i, (idx, row) in enumerate(cosine_sim_sample_data.iterrows()):
    sim_scores = cosine_similarity([row], cosine_sim_sample_data)
    similarity_matrix_sample.iloc[i] = sim_scores[0]

# Funcion para obtener recomendaciones de juegos
def get_recommendations_sample(title, similarity_matrix=similarity_matrix_sample):
    idx = sample_data.index[sample_data['Title'] == title].tolist()[0]
    sim_scores = list(enumerate(similarity_matrix.loc[title]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    game_indices = [i[0] for i in sim_scores]
    return sample_data['Title'].iloc[game_indices]

print(sample_data)
game_title = sample_data['Title'].iloc[998]
print(game_title)
recommended_games = get_recommendations_sample(game_title)
print(recommended_games)

# Le damos mas importancia a los numeros de reseñas en base a los tags populares y las características del juego
steamGamesData['Reviews_number'] = np.log1p(steamGamesData['Reviews_number'])
print(steamGamesData['Reviews_number'])

tag_columns = [f'tag_pc_{i}' for i in range(200)]
game_feature_columns = [f'Game_Features_Component_{i}' for i in range(50)]

steamGamesData[tag_columns] = steamGamesData[tag_columns] * 1.2
steamGamesData[game_feature_columns] = steamGamesData[game_feature_columns] * 1.4


# Funcion para que los titulos de los juegos en recommendations tengan todos la misma estructura entre comillas simples
def escape_quotes(recommendations):
    return recommendations.replace('"', "'")


chunk_size = 1000
recommendations_list = []

# Iteracion dentro de los juegos en grupos de 1000 ( chunk_size )
for start in range(0, len(steamGamesData), chunk_size):
    end = start + chunk_size
    data_chunk = steamGamesData[start:end].copy()
    data_chunk = data_chunk.dropna()


    cosine_sim_data = data_chunk.drop('Title', axis=1)
    similarity_matrix = cosine_similarity(cosine_sim_data)

    # Creamos las recomendaciones
    for i, title in enumerate(data_chunk['Title']):
        sim_scores = list(enumerate(similarity_matrix[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Exclude self
        game_indices = [i[0] for i in sim_scores]
        recommendations = data_chunk['Title'].iloc[game_indices].values.tolist()  # Convert to list

        # Aplicamos la funcion para que los datos se queden en el formato apropiado para la BBDD
        escaped_recommendations = escape_quotes(str(recommendations))
        # Creamos un dataFrame para el titulo con sus respectivas recomendaciones
        recommendations_df = pd.DataFrame({'Title': [title], 'Recommendations': [escaped_recommendations]})

        # Agregamos el dataframe a una lista
        recommendations_list.append(recommendations_df)

# Concatenamos todas las recomendaciones generadas en un solo DataFrame
final_recommendations_df = pd.concat(recommendations_list, ignore_index=True)

# Guardamos las recomendaciones en un CSV
final_recommendations_df.to_csv('recomendacion_juego.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

# Expresión regular para eliminar caracteres no alfanuméricos y no deseados (solo letras, números y espacios permitidos)
invalid_char_pattern = re.compile(
    r"[^a-zA-Z0-9\s]", flags=re.UNICODE  # Permitimos solo letras, números y espacios
)


def clean_text_with_re(text):
    # Eliminar caracteres no alfanuméricos ( la base de datos tiene que ser UTC8 )
    cleaned_text = remove_non_utf8(text)
    return cleaned_text

# Función para eliminar caracteres no UTF-8 (como los japoneses)
def remove_non_utf8(text):
    # Expresión regular para eliminar caracteres que no sean ASCII
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Leer el archivo CSV
input_file = 'recomendacion_juego.csv'
output_file = 'main_recomendacion_juego.csv' # Creamos en esta instancia

with open(input_file, mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = []

    # Leer la cabecera (primer fila) y agregarla al resultado
    header = next(reader)
    rows.append(header)  # Escribimos la cabecera primero

    # Leer cada fila, limpiar la lista de juegos y guardar los resultados
    for row in reader:
        try:
            # Usamos ast.literal_eval en lugar de eval para una conversión más segura
            cleaned_title = clean_text_with_re(row[0])  # Limpiar el título de cada fila

            game_names = ast.literal_eval(row[1])  # Convertir la cadena que representa la lista a una lista real
        except (ValueError, SyntaxError) as e:
            continue  # Saltar esta fila si hay un error al convertir

        cleaned_game_names = [clean_text_with_re(game) for game in game_names]
        rows.append([cleaned_title, cleaned_game_names])

# Escribir los datos limpios en un nuevo archivo CSV (usando UTF-8)
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print(f"Datos limpiados y guardados en {output_file}")

# Datos limpios del Dataframe entero ( archivo con el cual se pueden hacer mejoras a posterior, pero que ya no utilizaremos )
file_name = 'datos_listos_ML.csv'

steamGamesData.to_csv(file_name, index=False)
print(f'Data saved as {file_name}')