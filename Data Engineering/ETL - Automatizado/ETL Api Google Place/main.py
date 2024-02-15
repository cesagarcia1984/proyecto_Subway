import requests
import pandas as pd
from google.cloud import storage

def places_search_function(request):
    try:
        api_key = "AIzaSyC29wNEZ5ViqeN-gMiuulXJkz-Cs-t9XlY"
        query = "restaurant subway"
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&key={api_key}"

        all_results = []  # Almacenará todos los resultados

        while True:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Extraer los resultados de la consulta actual
                results = data.get('results', [])
                all_results.extend(results)  # Agregar los resultados actuales a la lista

                # Verificar si hay más páginas de resultados
                next_page_token = data.get('next_page_token')
                if not next_page_token:
                    break  # No hay más páginas, terminar el bucle

                # Esperar un breve período de tiempo antes de hacer la siguiente solicitud (para cumplir con las políticas de uso)
                import time
                time.sleep(2)

                # Configurar la URL para la siguiente página
                url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?pagetoken={next_page_token}&key={api_key}"
            else:
                return "Error en la consulta", 500

        # Crear un DataFrame con todos los resultados
        df = pd.DataFrame(all_results)

        # Guardar el DataFrame como un archivo CSV
        csv_data = df.to_csv(index=False)

        # Inicializar el cliente de Cloud Storage
        storage_client = storage.Client()

        # Obtener el bucket
        bucket_name = 'bucket_proyectosubway_1'
        bucket = storage_client.bucket(bucket_name)

        # Definir el nombre de la carpeta dentro del bucket
        folder_name = 'API_Google_Maps'

        # Definir el nombre completo del archivo incluyendo la carpeta
        file_name_with_folder = f'{folder_name}/subway_places.csv'

        # Subir el archivo CSV al bucket con el nombre de la carpeta incluido
        blob = bucket.blob(file_name_with_folder)
        blob.upload_from_string(csv_data, content_type='text/csv')

        return 'Datos guardados en Cloud Storage con éxito', 200
    except Exception as e:
        return f"Error en la función: {str(e)}", 500
