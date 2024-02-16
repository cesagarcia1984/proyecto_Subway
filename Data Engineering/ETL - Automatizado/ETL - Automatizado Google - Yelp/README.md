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
    <li><a href="#Tecnologías">Tecnologías Utilizadas</a></li>
    <li><a href="#Data-Lake">Data Lake</a></li>
    <li><a href="#Cloud-Function">Cloud Function</a></li>
    <li><a href="#Data-Warehouse">Data Warehouse</a></li>
    <li><a href="#Video">Video</a></li>
  </ol>
</details>

## Pipeline

Para el proceso de extracción, transformación y carga de los datos de este proyecto, se siguió el flujo a continuación:

![Pipeline_Automatizado]()


## Data Lake

El primer componente de nuestra estructura fue la creación de un Data Lake utilizando Cloud Storage de GCP. En esta etapa, realizamos la carga manual de los datos crudos en el Data Lake. Esto nos permitió almacenar grandes volúmenes de datos de manera segura y asequible, manteniendo su integridad y disponibilidad para futuros análisis.

![datalakepng]()


## Cloud Function

Para automatizar el proceso ETL, implementamos Google Cloud Functions. Configuramos un activador que se desencadena automáticamente cuando se realiza una carga manual de datos en Cloud Storage. Este activador inicia el proceso ETL, que incluye la validación, transformación, limpieza y carga de datos en nuestro Data Warehouse.

![cfuncpng]()


## Data Warehouse

El componente central de nuestra estructura de datos es BigQuery. Utilizamos BigQuery como nuestro Data Warehouse, donde se almacenan y gestionan todos los datos procesados después del ETL. Esta plataforma escalable y de alto rendimiento nos permite ejecutar consultas complejas y obtener resultados rápidos, lo que facilita la extracción de información valiosa.

![bigquerypng]()


## Video

En el video a continuación, podrás observar cómo se activa la función del ETL al cargar un archivo en nuestro Data Lake. Como resultado de este proceso, los datos procesados se almacenan en nuestro Data Warehouse. Te invitamos a hacer clic en el logotipo de YouTube a continuación para visualizar la demostración.

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
