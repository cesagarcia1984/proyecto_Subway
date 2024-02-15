import pandas as pd
import requests
import re
import os
from pandas import json_normalize
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
        df = pd.read_parquet(f_path, dtype=str)

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
        df_user_1 = df.iloc[:, :11]
        df_user_1 = df_user_1.drop(columns=['fans','elite','useful','funny','cool'])
        df_user_1['yelping_since'] = pd.to_datetime(df_user_1['yelping_since'])
        # Convierte la columna 'Date' a tipo datetime y Extrae solo la parte de la fecha
        df_user_1.loc[:, 'yelping_since'] = pd.to_datetime(df_user_1['yelping_since']).dt.date
        # Eliminar filas duplicadas completas y conserva la primera aparición
        df_user_2 = df_user_1.drop_duplicates()
        
        # Filtramos reviews
        # <<<< Filtrar por columna user_id de solo los usuarios que comentan >>>>
        project_id = 'sacred-result-412820'  # Reemplaza con tu ID de proyecto de GCP
        dataset_id = 'dt_y_user' # Reemplaza con tu ID de dataset en BigQuery
        filename = 'tb_usuarioscomentan'
        # Crear cliente de BigQuery
        client = bigquery.Client(project=project_id)
        table_id = f'{project_id}.{dataset_id}.{filename}'
        # Obtener DataFrame desde BigQuery
        query = f'SELECT * FROM `{table_id}`'
        df_usuarios = client.query(query).to_dataframe()

        # Filtramos reviews
        df_user_2 = df_user_2[df_user_2['user_id'].isin(df_usuarios['user_id'])]
        # Eliminar filas duplicadas basadas en user_id y conservar la primera aparición
        df_user_2.drop_duplicates(subset='user_id', keep='first', inplace=True)

        #==========================================================================================================
        df_user_2.to_csv("tb_y_user.csv", index = False) # Nombre de la tabla que se creará en BigQuery

        return df_user_2

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
        # df = df.astype(str)
        df['friends'] = df['friends'].astype(str)
        
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
            if main_folder == "yelp_user": # Nombre carpeta dentro del bucket 
                
                # Especificar
                project_id = 'sacred-result-412820' # ID del proyecto
                dataset = "dt_y_user."     # Nombre del dataset en big query
                table_name = "tb_y_user"  # Nombre de la tabla que se creara en BigQuery
                
                # crea el df segun el tipo de archivo
                data = leer_archivo(full_path)

                # llama la funcion para limpiar el df
                data_limpia = limpiar_df(data)
                
                # llama a la funcion para cargar el df
                cargar_df(project_id, dataset, table_name, data_limpia)

    except Exception as e:
        print(f"An error occurred: {e}")