import pandas as pd
import requests
import re
import os
from google.cloud import bigquery
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
        try:
            # Intentar leer el archivo json como si no tuviera saltos de linea
            df = pd.read_json(f_path)
        except ValueError as e:
            if 'Trailing data' in str(e):
                # Leer el archivo json conteniendo saltos de linea
                df = pd.read_json(f_path, lines = True)
            else:
                # Cualquier otro error
                print('Ocurrió un error cargando el archivo JSON:', e)

    # Revisar si el archivo es tipo parquet
    elif f_type == 'parquet':
        # Leyendo archivo en dataframe
        df = pd.read_parquet(f_path)
        print('Leyo correctamente el parquet')

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
        #================= ETL =================================================================================
        df1 = df.iloc[:, :14]  # Primer DataFrame con las primeras 15 columnas
        df2 = df.iloc[:, 14:]  # Segundo DataFrame con el resto de las columnas        
        df1 = df1.combine_first(df2)
        df_subway = df1[df1['name'].str.contains('subway', case=False, na=False) | df1['name'].isnull()]
        # Filtrar solo estados que analizaremos
        df_filtrado = df_subway[df_subway['state'].isin(['PA', 'FL','TN'])]
        # Renombrar columna de estado corto
        df_filtrado.rename(columns={'state': 'short_state'}, inplace=True)
        df_filtrado['review_count'] = df_filtrado['review_count'].astype(int)
        df_filtrado['latitude'] = df_filtrado['latitude'].astype(str)
        df_filtrado['longitude'] = df_filtrado['longitude'].astype(str)
        #Crear columna de stados
        mapeo_estados = {
            'PA': 'Pennsylvania',
            'FL': 'Florida',
            'TN': 'Tennessee'
        }
        # Aplicar el reemplazo de nombres 
        df_filtrado['state'] = df_filtrado['short_state'].map(mapeo_estados)
        df_filtrado.drop(columns=['attributes', 'hours','is_open'], inplace=True)

        #==========================================================================================================
        df_filtrado.to_csv("tb_y_business.csv", index = False) # Nombre de la tabla que se creará en BigQuery

        return df_filtrado

    except Exception as e:
        print(f"An error occurred: {e}")

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
        df = df.astype(str)

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
            if main_folder == "yelp_business": # Nombre carpeta dentro del bucket 
                
                # Especificar
                project_id = 'sacred-result-412820' # ID del proyecto
                dataset = "dt_y_business."     # Nombre del dataset en big query
                table_name = "tb_y_business"  # Nombre de la tabla que se creara en BigQuery
                
                # crea el df segun el tipo de archivo
                data = leer_archivo(full_path)

                # llama la funcion para limpiar el df
                data_limpia = limpiar_df(data)
                
                # llama a la funcion para cargar el df
                cargar_df(project_id, dataset, table_name, data_limpia)

    except Exception as e:
        print(f"An error occurred: {e}")