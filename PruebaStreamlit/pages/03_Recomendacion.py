import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from google.cloud import bigquery
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import LatentDirichletAllocation
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# =========================================================
# Cargar variables de entorno desde el archivo .env
load_dotenv()
# Obtener el valor de la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
credentials_path = "sacred-result-412820-2396f9ae3f21.json"
# Establecer la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
# ========================================================

st.markdown("## Clasificación y recomendación de locales Subway - desde Big Query")
st.markdown("### Clasificación de temas más discutidos")

# Credenciales al proyecto
project_id = 'sacred-result-412820'
client = bigquery.Client(project = project_id)

# Google reviews
g_dataset_id = 'dt_g_review'
g_table_id = 'tb_google_review'
g_dataset_metadata = 'dt_g_metadata'
g_table_metadata ='tb_google_metadata'

# Consulta para combinar las tablas y obtener los datos necesarios
query = f"""
SELECT 
    A.user_id, A.time, A.year, A.stars, A.text, A.gmap_id, A.sentiment_analysis, A.state, A.short_state, B.city
FROM 
    `{project_id}.{g_dataset_id}.{g_table_id}` AS A
JOIN 
    `{project_id}.{g_dataset_metadata}.{g_table_metadata}` AS B 
ON 
    A.gmap_id = B.gmap_id
"""
# Ejecutar la consulta y convertir los resultados en un DataFrame
df_combined = client.query(query).to_dataframe()


# Cambiar el nombre de la columna 'time' a 'date'
df_combined = df_combined.rename(columns={'time': 'date'})

#================= FILTROS ==================================
# Opciones iniciales vacío
default_options = {
    "state": [],
    "city": [],
    "year": []
}

filters = {}

for column in default_options:
    filters[column] = st.sidebar.multiselect(column.capitalize(), default_options[column] + list(df_combined[column].unique()))

# Realizar la consulta
if st.sidebar.button("Aceptar"):
    # Aplicar los filtros al DataFrame solo cuando el botón es presionado
    filtros = pd.Series(True, index=df_combined.index)
    for column in filters:
        if filters[column]:  # Aplica filtro si la lista no está vacía
            filtros = filtros & df_combined[column].isin(filters[column])

    # Filtrar el DataFrame
    df_filtrado = df_combined[filtros]

    #============= Preprocesamiento de texto ====================
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text, stop_words):
        try:
            # Check if the text is not None
            if text is not None:
                # Check if the text is a string
                if isinstance(text, str):
                    # Tokenize and remove stop words
                    tokens = word_tokenize(text.lower())
                    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                    return ' '.join(tokens)
                else:
                    # If not a string, return an empty string or handle it according to your needs
                    return ''
            else:
                # If text is None, return an empty string or handle it according to your needs
                return ''
        except Exception as e:
            # Handle any exceptions that might occur during text preprocessing
            print(f"An error occurred during text preprocessing: {e}")
            return ''

    df_filtrado['clean_text'] = df_filtrado['text'].apply(preprocess_text,stop_words=stop_words)

    #==================MODELO========================
    # Filtrar comentarios con sentimiento positivo
    positive_comments = df_filtrado[df_filtrado['sentiment_analysis'] == '1']
    negative_comments = df_filtrado[df_filtrado['sentiment_analysis'] == '0']

    # Calcular el promedio de las calificaciones de estrellas para cada local
    average_stars_positive = positive_comments.groupby('gmap_id')['stars'].mean().reset_index()
    average_stars_negative = negative_comments.groupby('gmap_id')['stars'].mean().reset_index()

    # Identificar el local mejor calificado (mayor promedio de estrellas entre comentarios positivos)
    best_rated_local = average_stars_positive.loc[average_stars_positive['stars'].idxmax()]
    worst_rated_local = average_stars_negative.loc[average_stars_negative['stars'].idxmin()]

    # Filtrar comentarios para los locales mejor y peor calificados
    #best_comments = positive_comments[positive_comments['gmap_id'] == best_rated_local['gmap_id']]['clean_text']
    #worst_comments = negative_comments[negative_comments['gmap_id'] == worst_rated_local['gmap_id']]['clean_text']

    # Eliminar filas donde la columna 'text' esté vacía
    df_filtrado = df_filtrado.dropna(subset=['text'])

    # Definir las listas de palabras clave relacionadas a las características
    lista_words_service = ['service', 'attention', 'friendly', 'courteous', 'prompt']
    lista_words_price = ['price', 'expensive', 'cheap', 'affordable', 'cost']
    lista_words_food = ['food', 'taste', 'quality', 'delicious', 'flavor']
    lista_words_payment = ['payment', 'pay', 'card', 'cash', 'transaction']

    # Identificar comentarios relacionados a cada característica para el local mejor calificado
    best_service_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_service))]
    best_price_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_price))]
    best_food_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_food))]
    best_payment_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_payment))]

    # Identificar comentarios relacionados a cada característica para el local peor calificado
    worst_service_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_service))]
    worst_price_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_price))]
    worst_food_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_food))]
    worst_payment_comments = df_filtrado[df_filtrado['clean_text'].apply(lambda x: any(word in x for word in lista_words_payment))]

    # Obtener el gmap_id del local mejor y peor calificado
    best_rated_gmap_id = best_rated_local['gmap_id']
    worst_rated_gmap_id = worst_rated_local['gmap_id']

    #============== Primera grafica ===============================
    # Concatenar todos los comentarios en un solo DataFrame
    all_comments = pd.concat([best_service_comments, best_price_comments, best_food_comments, best_payment_comments,
                            worst_service_comments, worst_price_comments, worst_food_comments, worst_payment_comments])

    # Contar las ocurrencias de palabras clave en los comentarios
    service_counts = all_comments['clean_text'].apply(lambda x: sum(1 for word in lista_words_service if word in x))
    price_counts = all_comments['clean_text'].apply(lambda x: sum(1 for word in lista_words_price if word in x))
    food_counts = all_comments['clean_text'].apply(lambda x: sum(1 for word in lista_words_food if word in x))
    payment_counts = all_comments['clean_text'].apply(lambda x: sum(1 for word in lista_words_payment if word in x))

    # Crear un DataFrame para los recuentos de temas
    df_counts = pd.DataFrame({'Service': service_counts.sum(),
                            'Price': price_counts.sum(),
                            'Food': food_counts.sum(),
                            'Payment': payment_counts.sum()}, index=[0])

    # Función para graficar un gráfico de barras horizontal
    def plot_horizontal_bar_chart(data, title):
        colors = plt.cm.Greens(np.linspace(0.2, 1, len(data.index)))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(data.index, data['Frecuencia'], color=colors)
        ax.set_title(title)
        ax.set_ylabel('Tema')
        st.pyplot(fig)

    # Ordena
    df_counts = df_counts.T
    df_counts.columns = ['Frecuencia']
    df_counts = df_counts.sort_values(by='Frecuencia', ascending=True)

    # Llamar a la función para graficar el gráfico de barras horizontal
    plot_horizontal_bar_chart(df_counts, 'Clasificación de temas más discutidos')

    #================== Impresion de detale ================
    # Obtener el 'rating' y 'city' asociados al gmap_id del mejor y peor local calificado
    best_rated_rating_city = df_combined[df_combined['gmap_id'] == best_rated_gmap_id][['stars', 'city']].iloc[0]
    worst_rated_rating_city = df_combined[df_combined['gmap_id'] == worst_rated_gmap_id][['stars', 'city']].iloc[0]

    # Crear DataFrames para los detalles de los locales mejor y peor calificados
    # Crear DataFrames para los detalles de los locales mejor y peor calificados
    df_best_details = pd.DataFrame({
        'Local': ['Mejor Calificado'],
        'gmap_id': [best_rated_gmap_id],
        'Rating': [best_rated_rating_city['stars']],
        'City': [best_rated_rating_city['city']],
        'Característica': ['Servicio' if len(best_service_comments) > len(best_price_comments) and len(best_service_comments) > len(best_food_comments) and len(best_service_comments) > len(best_payment_comments)
                            else 'Precio' if len(best_price_comments) > len(best_service_comments) and len(best_price_comments) > len(best_food_comments) and len(best_price_comments) > len(best_payment_comments)
                            else 'Comida' if len(best_food_comments) > len(best_service_comments) and len(best_food_comments) > len(best_price_comments) and len(best_food_comments) > len(best_payment_comments)
                            else 'Medios de Pago']
    })

    df_worst_details = pd.DataFrame({
        'Local': ['Peor Calificado'],
        'gmap_id': [worst_rated_gmap_id],
        'Rating': [worst_rated_rating_city['stars']],
        'City': [worst_rated_rating_city['city']],
        'Característica': ['Servicio' if len(worst_service_comments) > len(worst_price_comments) and len(worst_service_comments) > len(worst_food_comments) and len(worst_service_comments) > len(worst_payment_comments)
                            else 'Precio' if len(worst_price_comments) > len(worst_service_comments) and len(worst_price_comments) > len(worst_food_comments) and len(worst_price_comments) > len(worst_payment_comments)
                            else 'Comida' if len(worst_food_comments) > len(worst_service_comments) and len(worst_food_comments) > len(worst_price_comments) and len(worst_food_comments) > len(worst_payment_comments)
                            else 'Medios de Pago']
    })

    # Mostrar los detalles en forma de tabla en Streamlit
    st.write("### Indicador de locales mejor calificado")
    st.write(df_best_details)

    st.write("### Indicador de locales peor calificado")
    st.write(df_worst_details)