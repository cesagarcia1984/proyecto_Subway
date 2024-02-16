import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import os
from google.cloud import bigquery
from dotenv import load_dotenv
#pip install google-cloud-bigquery
#pip install python-dotenv
#pip install db-dtypes


# ===========================
# Cargar variables de entorno desde el archivo .env
load_dotenv()
# Obtener el valor de la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
credentials_path = "sacred-result-412820-2396f9ae3f21.json"
# Establecer la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
# ===========================

st.markdown("## Análisis de sentimiento - desde Big Query")

# Credenciales al proyecto
project_id = 'sacred-result-412820'
client = bigquery.Client(project = project_id)

# Yelp reviews
dataset_id = 'dt_y_review'
table_id = 'tb_y_review'
# Google reviews
g_dataset_id = 'dt_g_review'
g_table_id = 'tb_google_review'

# Obtener datos de la tabla
table = client.get_table(f'{project_id}.{dataset_id}.{table_id}')
table2 = client.get_table(f'{project_id}.{g_dataset_id}.{g_table_id}')

# Consultar tabla Yelp Reviews
query_yelp = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
query_job_yelp = client.query(query_yelp)
df = query_job_yelp.to_dataframe()

# Consultar tabla GoogleMaps Reviews
query_Google = f"SELECT * FROM `{project_id}.{g_dataset_id}.{g_table_id}`"
query_job_google = client.query(query_Google)
df2 = query_job_google.to_dataframe()

# Convertir la columna 'text' a tipo str
df['text'] = df['text'].astype(str)
df2['text'] = df2['text'].astype(str)

# Filtrar comentarios positivos y negativos
positive_reviews_y = df[df['sentiment_analysis'] == '1']['text']
negative_reviews_y = df[df['sentiment_analysis'] == '0']['text']
positive_reviews_g = df2[df2['sentiment_analysis'] == '1']['text']
negative_reviews_g = df2[df2['sentiment_analysis'] == '0']['text']

# Imprimir los DataFrames para verificar la presencia de la columna 'text'
#print("Positive reviews from Yelp:\n", positive_reviews_y)
#print("Negative reviews from Yelp:\n", negative_reviews_y)
#print("Positive reviews from Google Maps:\n", positive_reviews_g)
#print("Negative reviews from Google Maps:\n", negative_reviews_g)

concatenated_df_pos = pd.concat([positive_reviews_y, positive_reviews_g])
concatenated_df_neg = pd.concat([negative_reviews_y, negative_reviews_g])

# Filtrar valores nulos en la columna 'text'
df_pos = concatenated_df_pos[concatenated_df_pos.notnull()]
df_neg = concatenated_df_neg[concatenated_df_neg.notnull()]


st.markdown("### Nube de palabras (WordCloud)")

# Generar nube
def my_wordcloud(data, title):
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(' '.join(data))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)    
    
my_wordcloud(df_pos, 'Positivos')
my_wordcloud(df_neg, 'Negativos')