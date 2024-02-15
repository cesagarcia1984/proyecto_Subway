div align="center">

![wink](https://github.com/cesagarcia1984/proyecto_Subway/blob/5f07e6bbe88d6eb52bb98dc9f3717e792d9500fb/Imagen/Logo_DataStudio18.jpeg)
</div>

# ETL - Automatizacion de Google Places API 

## Indice

<details>
  <summary>Tabla de contenido</summary>
  <ol>
    <li><a href="#Flujo de Trabajo (Pipeline)">Flujo de Trabajo (Pipeline)</a></li>
    <li><a href="#Google Cloud Scheduler">Google Cloud Scheduler</a></li>
    <li><a href="#Google Cloud Functions">Google Cloud Functions</a></li>
    <li><a href="#Big Query - Data Warehouse">Big Query - Data Warehouse</a></li>
    <li><a href="#GIF">GIF</a></li>
    <li><a href="#Tecnologías">Tecnologías Utilizadas</a></li>    
  </ol>
</details>

## Flujo de Trabajo (Pipeline)
La obtencion de datos, por medios de la API, permite mejorar los procesos posteriores debido al flujo continuo de Datos:

![pipeline](https://github.com/claudiacaceresv/pf_yelp_google/blob/9e3091720adf4532c8996f0cbbd8c48bab716e92/src/Pipeline%20API.png)



## Google Cloud Scheduler
Generamos una tarea Programada, a traves de GCP Scheduler, la cual nos permite obtener Datos de la API Google Places. Esta tarea se ejecuta cada 24 horas, para obtener los datos de los Restaurantes 'Subway', activando una funcion de GCP Functions.


## Google Cloud Functions
Para extraer los datos, se creó una función en Cloud Functions (API-Place-Google) que por medio de una API Key, extrae los datos directamente desde la API de Google Places y almacena la información en BigQuery cada vez que se ejecuta la función.


## Big Query - Data Warehouse
El proceso de ETL devuelve una tabla llamada "Metadata", que se almacena en el Data Warehouse en Big Query dentro de un schema llamado "API_Google_Place", del cual se puede extraer la información con consultas SQL y conectándose al cliente de Big Query.


## GIF
En este GIF podra observar el proceso de llamado de la API a traves GC Scheduler y como se guarda la información en el Data Warehouse alojado Big Query.


## Tecnologías
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
- Cloud Functions
- Cloud Storage
- Big Query
- Cloud Scheduler

<div align="center">

![wink](https://github.com/claudiacaceresv/pf_yelp_google/blob/ecae7e03590019bb0db5c0c61810e27e209499cb/src/Video%20ETL%20API.gif)

</div>