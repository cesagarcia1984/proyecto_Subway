import pandas as pd
import requests
import re
from functools import reduce
import numpy as np

def leer_archivo(f_path):
    """
    Lee el archivo según su extensión. Disparado por la funcion "captura_evento"
    Args:
        event (dict): Event payload.
        file_path (str): ruta del archivo
        file_type (str): tipo del archivo
    """
    # Extraer el tipo de archivo
    f_type = f_path.split('.')[-1]
    
    # Revisando si archivo es csv
    if f_type == 'csv':
        # Leyendo archivo en dataframe
        df = pd.read_csv(f_path)

    # Revisando si archivo es json    
    elif f_type == 'json':
        print('Se leyo correctamente1')
        try:
            # Intentar leer el archivo json como si no tuviera saltos de linea
            df = pd.read_json(f_path)
            print('Se leyo correctamente2')
        except ValueError as e:
            if 'Trailing data' in str(e):
                # Leer el archivo json conteniendo saltos de linea
                df = pd.read_json(f_path, lines = True)
                print('Se leyo correctamente3')
            else:
                # Cualquier otro error
                print('Ocurrió un error cargando el archivo JSON:', e)

    # Revisar si el archivo es tipo parquet
    elif f_type == 'parquet':
        # Leyendo archivo en dataframe
        df = pd.read_parquet(f_path)

    # Revisar si el archivo es tipo pkl (Pickle)
    elif f_type == 'pkl':
        try:
            # Leyendo archivo en DataFrame desde Google Cloud Storage
            df = pd.read_pickle(f_path)
        except Exception as e:
            print(f'Ocurrió un error al leer el archivo Pickle: {e}')
    
    return df

def limpiar_df(df):
    """
    Limpia el df "sales_count_month". Disparado por la funcion "captura_evento"
    Args:
        data (DataFrame): dataframe a limpiar.
    """
    
    try:
        #================= ETL ================================

        # Filtrar por estado
        patron_pa = r'\b(?:NY|New York|NYC)\b'
        # Filtra el DataFrame para incluir solo filas donde 'address' contiene la cadena 'NY' en cualquier forma escrita
        google_metadata_newyork = df[df['address'].str.contains(patron_pa, case=False, na=False, regex=True)]
        # Filtrar solo negocios de Subway
        google_metadata_newyork = google_metadata_newyork[google_metadata_newyork['name'].str.contains('subway', case=False, na=False) | google_metadata_newyork['name'].isnull()]
        # Reindexar el DataFrame después de filtrar
        google_metadata_newyork = google_metadata_newyork.reset_index(drop=True)
        # Normalizar nombres de Subway
        valores_a_normalizar = ['SUBWAY®Restaurants', 'Subway Restaurants', 'SUBWAY® Restaurants', 'SUBWAY']
        # Filtra las filas que contienen alguno de los valores específicos
        subway_rows = google_metadata_newyork['name'].isin(valores_a_normalizar)
        # Normaliza los valores específicos en la columna 'name'
        google_metadata_newyork.loc[subway_rows, 'name'] = 'Subway'
        # Filtramos solo Subway
        g_metadata_subway_newyork = google_metadata_newyork[google_metadata_newyork['name'] == 'Subway']
        # Eliminar duplicados en gmap_id
        g_metadata_subway_newyork = g_metadata_subway_newyork.drop_duplicates(subset='gmap_id', keep='first').reset_index(drop=True)
        # Normalizamos fechas largas 
        g_metadata_subway_newyork['address'] = g_metadata_subway_newyork['address'].apply(lambda x: x.rsplit(', United States', 1)[0].strip() if x.endswith(', United States') else x)
        # Separa la columna 'address para extraer datos de la ubicación
        g_metadata_subway_newyork = g_metadata_subway_newyork.copy()
        g_metadata_subway_newyork['address'] = g_metadata_subway_newyork['address'].str.split(',').str[1:].str.join(',').str.strip()
        # Extrae datos de la ubicación
        g_metadata_subway_newyork['city'] = g_metadata_subway_newyork['address'].str.split(',').str[1]
        g_metadata_subway_newyork['state_PostalCode'] = g_metadata_subway_newyork['address'].str.rsplit(',', n=1).str.get(-1).str.strip()
        # Dividir la cadena en código postal y resto
        g_metadata_subway_newyork[['short_state', 'postal_code']] = g_metadata_subway_newyork['state_PostalCode'].str.rsplit(n=1, expand=True)
        # Filtrar dígitos para obtener el código postal
        g_metadata_subway_newyork['postal_code'] = g_metadata_subway_newyork['postal_code'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
        # Filtrar solo el estado deseado
        g_metadata_subway_newyork = g_metadata_subway_newyork[g_metadata_subway_newyork['short_state'] == 'NY']
        # Eliminar columnas innecesarias
        g_metadata_subway_newyork.drop(columns=['state','url','description','hours','address','state_PostalCode', 'price'], inplace=True)
        # Crea la nueva columna 'state' con el valor 'New York'
        g_metadata_subway_newyork['state'] = 'New York'

        #========================================================
        g_metadata_subway_newyork.to_csv("tb_google_metadata.csv", index = False) # Nombre de la tabla que se creará en BigQuery
        print('Se guardo correctamente')
        print(type(g_metadata_subway_newyork))

        return g_metadata_subway_newyork

    except Exception as e:
        print(f"An error occurred funcion limpiar: {e}")

def cargar_df(project, dataset, table, df):
    """
    Carga el df limpio en bigquery. Disparado por la funcion "captura_evento"
    Args:
        project_id (str): nombre del proyecto
        dataset (str): ubicacion del dataset de destino en bigquery
        table_name (str): nombre de la tabla de destino en bigquery
        data_limpia (DataFrame): dataframe limpio para cargar a bigquery
    """
    
    try:
        # convierte todo el dataset a str para almacenar
        # df = df.astype(str)
        df['category'] = df['category'].astype(str)
        df['MISC'] = df['MISC'].astype(str)
        df['relative_results'] = df['relative_results'].astype(str)
        
        # guarda el dataset en una ruta predefinida y si la tabla ya está creada la reemplaza
        df.to_gbq(destination_table = dataset + table, 
                    project_id = project,
                    table_schema = None,
                    if_exists = 'append',
                    progress_bar = False, 
                    auth_local_webserver = False, 
                    location = 'us')
            
    except Exception as e:
        print(f"An error occurred: {e}")

def captura_evento(event, context):
    """
    Triggered by a change to a Cloud Storage bucket.
    Args:
        event (dict): Event payload.
    """

    try:
        # Obteniendo ruta de archivo modificado y tipo de archivo
        file_bucket = event["bucket"]
        file_path = event['name']
        file_name = file_path.split('/')[-1].split('.')[-2]
        full_path = 'gs://' + file_bucket + '/' + file_path
        
        # Ejecuta el código si los archivos se cargan en la carpeta correcta del bukcet
        if '/' in file_path:
            main_folder = file_path.split('/')[0]

            # Especifica el conjunto de datos y la tabla donde va a almacenar en bigquery
            if main_folder == "google_metadata_nuevayork": # Nombre carpeta dentro del bucket 
                
                # Especificar
                project_id = 'sacred-result-412820' # ID del proyecto
                dataset = "dt_g_metadata."     # Nombre del dataset en big query
                table_name = "tb_google_metadata"  # Nombre de la tabla que se creara en BigQuery
                
                # crea el df segun el tipo de archivo
                data = leer_archivo(full_path)

                # llama la funcion para limpiar el df
                data_limpia = limpiar_df(data)
                
                # llama a la funcion para cargar el df
                cargar_df(project_id, dataset, table_name, data_limpia)

    except Exception as e:
        print(f"An error occurred: {e}")