<div style="text-align: center;">
  <img src="https://github.com/cesagarcia1984/proyecto_Subway/blob/5f07e6bbe88d6eb52bb98dc9f3717e792d9500fb/Imagen/Logo_DataStudio18.jpeg" style="width: 100%;" alt="wink">
</div>

# Desarrollo de ETL automatizado

## Índice
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenido</summary>
  <ol>
    <li><a href="#Pipeline">Pipeline</a></li>
    <li><a href="#Data-Lake">Data Lake</a></li>
    <li><a href="#Cloud-Function">Cloud Function</a></li>
    <li><a href="#Data-Warehouse">Data Warehouse</a></li>
    <li><a href="#Video">Video</a></li>
    <li><a href="#Tecnologías">Tecnologías Utilizadas</a></li>
  </ol>
</details>

## Pipeline

En la siguente imagen se puede observar los pasos que se siguen para la Extraccion, Transformacion y Carga de los datos:

![Pipeline_Automatizado](https://github.com/cesagarcia1984/proyecto_Subway/blob/f7c045ca53065ae25fe2075a9330e6143417ae17/Imagen/Pipeline_Automatizado.PNG)


## Data Lake

La generacion del Data Lake se inicio con la ingesta manual de los Dataset, posterior a su transformacion. Se cargaron en el Service de Google Cloud Storage, en el Bucket generado. Los Dataset que fueron ingresados para formar el Data Lake son los descargados de Google Maps y de Yelp.

![datalakepng]()


## Cloud Function

Posterior a la creacion manual del Data Lake, se utilizo el servicio Google Cloud Function, en el cual se generaron varias funciones que permiten el procesamiento de manera Automatica, cada vez que ingresa un nuevo archivo al Data Lake. Esto se pudo lograr a traves de la creacion de 'Triggers', disparadores que una vez activados comienzan con la transformacion automatica de los archivos para enviarlos al servicio de Big Query.

![cfuncpng]()


## Data Warehouse

Utilizamos BigQuery como nuestro Data Warehouse, donde se almacenan y gestionan todos los datos procesados después del ETL. Esta plataforma escalable y de alto rendimiento nos permite ejecutar consultas complejas y obtener resultados rápidos, lo que facilita la extracción de información valiosa.

![bigquerypng]()


## Video

En el video a continuación,se puede observar como se produce el proceso de ETL de manera Automatica. Desde el ingreso del dato en el Data Lake, su transformacion por medio de las funciones como su paso a constituir el Data Warehouse.

<div align="center">
  
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)]()
  
</div>

## Tecnologías
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
- Cloud Storage
- Cloud Functions
- Big Query
