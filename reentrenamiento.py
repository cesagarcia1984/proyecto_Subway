import pandas as pd
import joblib
import streamlit as st
from surprise import Dataset, Reader

# Cargar el DataFrame
file_path = 'D:\\Users\\Cesar\\Desktop\\Proyecto_Final_Grupal\\ML\\google_unificado_2.csv'
df = pd.read_csv(file_path)

# Seleccionar las columnas relevantes
df = df[['user_id', 'address', 'rating']]

# Configurar el lector de Surprise
reader = Reader(rating_scale=(1, 5))

# Crear el conjunto de datos de Surprise
data = Dataset.load_from_df(df, reader)

# Dividir el conjunto de datos en entrenamiento y prueba
trainset = data.build_full_trainset()


# Crear y cargar el modelo desde el archivo
model_filename = 'C:\\Users\\Cesar\\OneDrive\\Escritorio\\entorno_virtual\\myenv\\modelo_recomendacion.joblib'
loaded_model = joblib.load(model_filename)

# Función para obtener las 10 mejores recomendaciones para un user_id dado
def get_top_recommendations(user_id):
    # Obtener las predicciones para el usuario dado
    user_predictions = []
    for item_id in df['address'].unique():
        user_predictions.append(loaded_model.predict(user_id, item_id).est)

    # Crear un DataFrame con las predicciones y las direcciones correspondientes
    recommendations_df = pd.DataFrame({
        'address': df['address'].unique(),
        'prediction': user_predictions
    })

    # Ordenar por predicciones en orden descendente y seleccionar las 10 mejores recomendaciones
    top_recommendations = recommendations_df.sort_values(by='prediction', ascending=False).head(10)

    return top_recommendations['address'].tolist()

# Configurar la aplicación Streamlit
st.title('Recomendador de Direcciones')

# Obtener el user_id de la entrada del usuario
user_id_input = st.number_input('Ingrese el user_id:', min_value=df['user_id'].min(), max_value=df['user_id'].max())

# Mostrar las 10 mejores recomendaciones para el user_id dado
if st.button('Obtener Recomendaciones'):
    recommendations = get_top_recommendations(user_id_input)
    st.subheader('Top 10 Recomendaciones:')
    st.write(recommendations)
