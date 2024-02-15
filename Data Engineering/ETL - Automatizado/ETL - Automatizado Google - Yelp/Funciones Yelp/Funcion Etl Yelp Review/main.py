import pandas as pd
import requests
import re
import os
from pandas import json_normalize
from functools import reduce
import json 
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

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
        # Eliminar filas completas duplicadas
        df = df.drop_duplicates(keep='first')
        # Convertir la columna 'date_usuario' de marcas de tiempo a formato legible
        df['date'] = pd.to_datetime(df['date'])
        # Crear columna con el año de la columna 'time' 
        df['year'] = df['date'].dt.year   
        # Extraer solo la fecha sin hora
        df['date'] = df['date'].dt.date
        ## Convertir los tipos de datos
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['year'].astype('Int64')
        #Filtramos solo reviews desde el año 2015
        df_filtro = df[df['year'] >= 2015]
        # Filtrar solo locales de subway
        lista_locales_business_id = ['VZzivJEbmHItOxNXJB3SpA', 'vuL3QCjFZ7lU5LMdUAmZ9Q', 'gi90NMmRhzl8vzM83UNOVA', 'xxxL7I3hGCEymNlYLXUyBA', 'wXBAAafoiZ4Nty2ajuBtfw', 'CmDTNEVReN6CgE6ctlZhag', 'yDriUh65Wpx1HnTmXln9nw', 'KE_oM4y1GMI3YMOnnqHqRQ', '31s1x27DnN2V-ptUWEdfTQ', '1QtF1bFgzG3Jo0-L4kv0Cg', 'HZtkTPfSshfRjpQR7x1Raw', 'zJdQwRbI7p0bH_60sfBSYA', '4KPEgM1E7WgQQqd0IAIopg', 'qpIpYSzySrqNmPWEPNplug', 'BQdg19sukvw90QJ4tbY-BA', 'DfkO7MHuy2wxgeUZ2-o3BA', 'qBvsNlxtqy8hPxx1C7HuVA', 'cdPjDZubPvok0VHv654Tvw', 'XF86J5bJKkmueBL-PmPbcg', '6Kc2FhLppRhWBXDKxjxj2A', 'vLdELbSYtwKVpkl4ownLEA', 'yheGRZWHC-RJoKlqepj0hw', 'NFdlTTZrmXejfKyz7v5T3Q', 'swKXtO22QjVIZTBU8UP7Pg', 'M2S5ELqHjz1B1booBTN1dQ', '67QczBNcTAYq8alxqwiZ5g', 'd-hMYlJ-hMAHSgd3KvYcWg', 'EwblShAzTJ4Op6aJZYYuVA', 'wQnVwt98pXX0n0Szro_XGg', '7ALZQc35zGG0a1SwSnExJg', 'AnS7cUS7RzXnEBx6c7y5JA', '_AvFiC-whJC9g9ndd3XGtA', 'Rx6eQL9PdxJK8wgCbKDu-A', 'wgDanVXxbMuP3djeFCk4PQ', 'q2rNovrR1ygAr5TLHSX70g', 'GwUfmvUOAWmByr0rRn2aTA', 'VqXABZYXe54vr5kpsKbFsg', 'ybgLYF_SpBTmlWqa3S3lcw', 'BQVgKbITz9nPnQASBjH63g', 'Yoxril34Timi9_e2nAvdHg', 'CyctXClhoy8YamJjcE2ifg', 'PFUgAn44e2JkK-NZiS0zrg', 'KlhTXr1S73X7xr-qQP8-Ww', 'k1XyeYNiUxNErHhuIIo_-A', 'u401RMGQdF4N5Q6omN385A', '0yN1rxCrEKqQIUk-nqyWVw', 'FDUFDCE5VML0zS4tSegjcg', 'rslVFy3PM5n-Mlv1Vvaq_A', 'rhiHVsxEpESpDjwd1xcJIg', 'NWjQ3f2b3cL7CGalFwwY_A', 'JbdGiAzMBRZdXIlDVISeBQ', 'oxRbrGrFxB2BOBLQca9c1g', 'yAcvEoMj064CYIXGF6FS0w', 'p5Vd4ag-1nxsCdmV0EEcDw', 'HDQGViGXTflhNta6Da1eZw', 'veiWJpG89acKOqKLknGeWw', 'UoWQQanrWuzXteBerAis2Q', '9o8RiObxO9L7gXzLHt-deA', 'Rp-LsUkZrQhxvEGPe9nhoQ', '-RS_7SU42foCTItx4f3C6A', 'Jnxn6VH6L0fwSZDuBTjpTA', 'asNmnXz9rT7JpJuvuX408w', '3031rErwJYYpk5IafsrFyw', 'zRjpagNQNZoovYfnmKxYmw', 'bndEHp-3hIlUVXmchrj3EA', 'MPQww2b74ASo1CW05BpubA', '0ihbRd1mnVKCP6h8TEpqng', 'UNyFn1P2-ViqdrucNYHPGA', '3p1gTXaVUKprXnzvhq7RRA', '-BLKZfw-FX7602K59OpBgg', 'dYWnqvJZnECiikUqyH9VuQ', 'bJ7_F_3uesqOPvSK8W8TxQ', '-TvJ8KJdXus3_o6uvG2GHQ', 'KrXTA30iqdvb7iNIYdFCVQ', 'An_QUi1YfPUf5kXJEgZMeQ', 'Ov-61MSpC7et_KKbsepcyQ', 'E1GEVyZCCdmT2zyp4vx2wA', 'oxlvgv2HBuPmpNlVHru0yQ', '6m_4mndX68YhS7daKo_5_g', 'RSPBRkjkHcjcR_KrVimBBQ', 'PdnqUb4snxEV8TG6gNP1_Q', 'TdMja0lg8l5kdTc9qtZoZg', 'SZnRcUzK_HB-UTKpwU-dfg', 'EvDHYUSTMma0jQdTROMD5Q', 'mwRqQENSKmBIGZEfWRzX5g', 'XFQ2gs8V3rMGOAENseXJIA', '8OyIu5q6Jc_pCeu7vUZXBw', 'JXNvb6-lBFhxowZ_y7QHlA', 'WIwRxBPvL8Md-EeZapgtiw', 'RatJCtQ8j4cG80vAPr9UAw', '1lwulkAD5htni3rbHW8oeg', 'ZoYq8ywyVBqBjY2uk61W8w', 'lcTR-NewLIrTTYwyAJJzUg', '9_DJMSzn5PtMit78KFrwaw', 'XXPr8iq_HsUepa7puBxc_g', '-Xzvm1LxWyO7QZu9Q0UcSw', '-iclF-OurSdeL0qcHhXuww', 'bQg9cm8xQ-8OmsIZn39V8Q', 'fq2r7qCmyxOQ0NHPRjOhlw', 'iTYC23Iusj3pnw-P7EFXGw', 'ei2lOmJVzOsjCV6PR4jVig', 'dM8VFFJsOCe2gN9pNBij1w', 'lFuja-ynvWAMLOqgw4Hzdw', 'MFAXAqQoJWUr8I5_nuGuuA', 'nP0Bf2hdJiw8kQmm9MgiqQ', '7e4PHzKqvtxcbxcXy6Jn3g', 'iDhp4LM3no0BodRzm4inCA', '0VSWfyUe9o-8rWAHf84syg', 'PF21rnDm1-rqhiNYSnszFA', 'lXVc6Ogts0cTH4jsJ5CIQg', 'rwZ-1fH9vdh1KRAowovXOQ', 'Ys9paLTObAmjju3zqF11_Q', 'wqRg8cvjxHTi1zIAOrLZ8w', 'YwcMPegVx8JLrNQxN2ynog', '2fSwH6Ga9BJcjFpp7F-baw', 's5I05cRXMRgeajFVKuu4oQ', '3THqdvjsEoodk6g0jJNN_g', 'w1gBmPNdTcRFVUKhnhIu3A', 'ZCJtBGZ3uZlXSgKGUMVo-w', '1NBOCuiFeG6DAkR_dd6pTA', 'rbsvykOaueH9cJES_ZHmUQ', '3rX3M3Oc4j-bN9jyX_TDbg', 'YuuKn_M5k488_23N0IVwWA', '2qg67Mq6cKjl1GPbls32Dg', 'y98y26WYSZqYCLXH-uQRLA', 'C_2agQxDB4GMNB-iG6Reeg', 'QVE3HLrEsRfMmOBvFJcExg', 'gK-uICnCY83BL8nzmSKyRg', 'G_1qwUSVAunua4Eiznj0hQ', 'oZuwJ73TOT2hsKXozAvckA', 'AnHSEz3G2AL4GK6c1-6xew', 'Om6OwVsJ_t2XHQBYKnlcHg', 'ijRJToHha_hzjSw3aod9HQ', '0xKA6W8zTfs9MH-P-yvYNg', 'A3FD_WGaUBASqbR8ID6ZUA', 'gwnS2dSQ8O7z3tyAZgGd1Q', 'hA0h72kDWsguR01u00YUgQ', 'A3xLUQ70z8Vjsr-IeUdlWQ', 'IQTT8y_QtumOscnVgZvprQ', 'dRYfyAnwZGfDKs83fOXotg', 'kEi05FoJJALcbz31dxmDEw', 'gIENOFOP1slo5UsrZRqEOw', 'vnB_NyVjpp705jDEypH2vA', 'NFB1Q0zyy6LS9eraHp-N5A', '4OxrTuQqlqSptzDF5aJuSA', '9xFg3NPiL9gLQZEY2AM9gQ', 'ux3DPp2_VDqWH1jO6Ic-vg', 'e9dMpY4Lm8w9z_A4br4Slg', '40yZy3NmbB9xjnFHOPdrag', 'C7DQfgxrhHUl5bYtC5eMAA', 'LnJVX8_vVq1QIlbEGQHYnw', '2ZwBQdlouprtw_MMOi9hxw', 'av45C-bRKQ-1-zTRTWNXng', 'X3ItPYxc6dIvFXp6Xij21w', 'b2kjgx_Py8_mC8xXkTnRIg', 'F5TgTlCw5Zl-AJGGT5v0ZQ', 'bz3Z6Lj3_r9BWAClE1cHpQ', 'VnECSKZnz-tDhDZ74s6k_w', 'I1TUWTWvA5X-fTwGfuwX4A', 'fra6poZMqkjuU0Gvhpb7hg', 'OtFQjONNg0z6TdSQ1Zxb9w', 'XFUuPGGzap22YQY_denOVQ', 'tUVHyBR-i6Jzrtp8FAiROA', 'aXcjf0vL1U4FmWANWyf1nQ', '-7UDKbg_8TL4LVuYR6Ooyw', 'bAoaqkzsAPJ8gZ6Ffw1kGg', 'GxaFs9X1Z_XbS9KvahOSHg', 'Whr0BgfTnl0MHWEvM5mMJQ', 'W4eCcwkxWoy2-uV8-aFmlw', 'nSdX0u7XiCItLQNTtlsmUw', 'EEJJgQ69_ra2A7DHWWsl5Q', 'CK961tD6ceTtZnPhXTxX8Q', 'WRphxfyuFw7EXdwAidFbRA', 'yMkuFECeuKOZi_4e-qPahA', 'gTrMIREEyNErXk0oGkXmlg', 'n2TGZRY4xTXASWh0baHzEw', '1FliMoYQnq72HRxEIPY0_A', 'kZNtf6Pw5sEohpQl_4a3AA', 'g9-krE9nt-pyIcKhuUO3-A', 'vpV8g_SAY8TR8Jmzevuw5g', '78X2nrDyQSjO4-VgPI9X2w', 'lMNB-M0rzOBZfhlmAUeLCA', 'hn3-6Oa6AzjZ-VmZ9BUw2A', '2HLZfbL-6lcr9jhriW6GeA', 'gaa1fQD2ts9pNL2tvHXLOA', 'np5GqYxPjiwGNPSlR-V_iA', 'lW0TV2wz00jSbtNV4JqA0Q', 'VtGfTOQvs2n0L18JHGNkgQ', 'Fua2G2W-oE7XoWnNIbnE5A', 'nw58qNTeh7gRyJsEZv9EIg', 'cuCa48riHFi_YYEZFrSApw', 'eLS3G2FMhq1kJ5obGYsG0Q', 'Pz49gFAaCXB130Q7fhwtmA', 'CSiJiPKT_OOJ6CzB0R2_GQ', 'JkkazxGDTPQV-9670qxx3w', 'FXL5bNO7W9_xPpEhkynnYQ', '6ABon6uRYt9Ji_W4ehn9lg', '3VdScOpQ4N1N-EqjA9NwZg', '4iRZPPkZ4-EDbgEoYswUMw', 'H3G4upTI99ZKxtjLkDXz7Q', 'jzJFNXrHorEnv0FqtB0fTQ', 'FR1xz_cCRWHGn6gZHa1K9g', 'FLYfQFkSxC5sFOkq_nb8Cw', 'YHpaAneR3gxFm6Ol_EKJAg', 'mca8Sx8I9M9Jz75O9f096Q', '5CW82H2eIcCAH5AGzF7clQ', 'phlcnA8qZlthpAu1YLOw2A', 'qqBPf0WQf8EUahAuO9oN1A', 'LB9agaVVjs20u-mEYXLv3Q', 'gB33WKhCqPpWvJeUWcHlvg', 'ZJQZ0QLow05pfXLjzsjm8w', 'qhNIJZBstJWVg63SWtus0g', 'PQaHCaL-EUtWmTiejAQ30Q', '4zy6I_nO7u4FVo1305mSfg', '2oB29blmV7_i5kR5GvDxeQ', 'UwDYz4wwqjr_xqhuW0bM0A', 'ZR0djeQOJnCo5MezS6giow', '9JZ2DY3PNLzqQtW2GkQUCg', 'Raj6Hh5PYXS9cVRPAksEsQ', 'CLXAH-PtMswVupgkEsW3uQ', '2mdQ6nhqreMnkFkSJuY6NA', 'FjVAwrTr0AMfs78pDLamtA', 'pX_IVV96CreozyPtKNyEGg', '4z-7YMw9_YMszg2oETuKbw', '1W4gz758cCBQHEmlCAycPA', 'u4FJ7DgSX2rTzWT3A9gIKQ', 'GXwChmM3Kam95-BWU5nCxg', '5n9LmVHeAFsfWJv520et0w', 'L617WGpANJZW8orvCdOO1w', 'qJsbhljm1ccXy5H826o6Og', 'tZPmtEI7zFaCPyhpy-mrHg', 'Yn6YpvotFLYofHSHEqwOiw', 'uVntIsdb-gMdmrCY7bvnBw', 'WMyF7FDJBl3H2cL0Pe25Lg', 'SqRS8tgUVzRWV2FgQ0A5PQ', 'hwRGsA-nW5Hi9LTApwA5RA', 'k0PzT0_WiOZXndrsz8fIxw', '4eFOGnHxVgVet0gUSF7cXg', 'F0QWKYUmBkxWOap1xspVzw', 'O0e34ICtjtOZuvSN6-lGxg', 'kGiXSCnUl6-LMJ4FLn2k4Q', 'uYyfBdh2-v4Lg165p9UjNQ', 'SviBUV-0nWV9dpS1yxDuTw', 'YnAlDbAGvBD8c4oSO8BVKA', 'bP1lwYkGwLvw0TqhuamnSA', 'g88mb6am4W2u_XNDN00y1A', '-x-niT6JolhXi6VuKclgxw', 'rpeLkHUD7iyPPoTessBPMA', '6dBTa5IQLANpgRaWZJCMBQ', 'iZjH7SKfiPSplPIHvbhwtQ', 'GFrT38hSaIZ2xO6WCV3rbg', 'cdbEEL_0TLOPgdfsB3PUvA', 'C96ynw3G1KrqBOdafxbTKw']
        df_filtro = df_filtro[df_filtro['business_id'].isin(lista_locales_business_id)]
        # Convierte la columna 'Date' a tipo datetime y Extrae solo la parte de la fecha
        df_filtro.loc[:, 'date'] = pd.to_datetime(df_filtro['date']).dt.date
        #===============Analisis sentimiento========================
        def limpiar_texto(texto):
            if isinstance(texto, str):
                texto = texto.lower()
                texto = re.sub(r'[^a-z0-9\s]', '', texto)
            return texto

        df_filtro['text'] = df_filtro['text'].apply(limpiar_texto)

        # Inicializar SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        # Función para realizar el análisis de sentimientos y asignar categorías
        def categorize_sentiment(review):
            # Verifica si la reseña está ausente
            if pd.isnull(review) or not isinstance(review, str):
                return 1  # Valor por defecto si la reseña está ausente
            else:
                # Realiza análisis de sentimientos con SentimentIntensityAnalyzer
                sentiment_score = sia.polarity_scores(review)['compound']
                
                # Asigna categoría según la escala proporcionada
                if sentiment_score <= 0:
                    return 0  # Negativo
                else:
                    return 1  # Positivo

        # Aplicar la función a la columna 'review' y crear una nueva columna 'sentiment_category'
        df_filtro['sentiment_analysis'] = df_filtro['text'].apply(categorize_sentiment)
        # Convierte la columna 'sentiment_analysis' a tipo de datos object
        df_filtro['sentiment_analysis'] = df_filtro['sentiment_analysis'].astype('object')
        # Reemplaza 'SD' en la columna 'sentiment_analysis' donde 'text' sea None
        df_filtro.loc[df_filtro['text'].isnull(), 'sentiment_analysis'] = 'SD'
        df_filtro['sentiment_analysis'] = df_filtro['sentiment_analysis'].astype(str)
        df_filtro['stars'] = df_filtro['stars'].astype(int)

        #==========================================================================================================
        df_filtro.to_csv("tb_y_review.csv", index=False) # Nombre de la tabla que se creará en BigQuery

        return df_filtro

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
            if main_folder == "yelp_review": # Nombre carpeta dentro del bucket 
                
                # Especificar
                project_id = 'sacred-result-412820' # ID del proyecto
                dataset = "dt_y_review."     # Nombre del dataset en big query
                table_name = "tb_y_review"  # Nombre de la tabla que se creara en BigQuery
                
                # crea el df segun el tipo de archivo
                data = leer_archivo(full_path)

                # llama la funcion para limpiar el df
                data_limpia = limpiar_df(data)
                
                # llama a la funcion para cargar el df
                cargar_df(project_id, dataset, table_name, data_limpia)

    except Exception as e:
        print(f"An error occurred: {e}")