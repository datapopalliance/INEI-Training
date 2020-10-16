# INEI-Training


## Introducción

Este repositorio contiene la plática de imágenes satelitales que se utilizó en el segundo día del training.

## Instalación

Para correr esta presentacion,  se debe crear un ambiente de [Conda](https://docs.conda.io/en/latest/) con version de *Python 3.7*.  Una vez activado el nuevo ambiente de Conda, se deben instalar las paqueterías necesarias para que corra el archvio .ipynb. Estas paqueterias se encuentran en el archivo *requirements.txt* y se instalan corriendo el siguiente comando en la terminal.



```bash
pip install -r requirements.txt
```



Nota: para que el comando anterior funcione, uno debe de estar situado dentro de la terminal, en la carpeta donde se encuentra el archivo requirements.txt



Una vez que se tiene el ambiente con las paqueterias necesarias, se puede ver el archivo dentro de jupyter corriendo el siguiente comando en la terminal



```bash
jupyter-lab
```



Para saber mejor como funciona jupyter-lab visite el siguiente [link](https://jupyterlab.readthedocs.io/en/latest/)



En caso de que se quiera ver el archivo .ipynb como presentacion, se deben instalar algunas extensiones de jupyter con el siguiente comando en la terminal



```bash
pip install jupyter_contrib_nbextensions
```



Despues, se debe de correr el siguiente comando en la terminal dentro de la carpeta donde se encuentra el archivo .ipynb



```bash
jupyter nbconvert Imagenes_satelitales_para_medir_los_Objetivos_de_Desarrollo_Sostenible_Parte_I.ipynb --to slides --post serve
```



## Otros Datos

como la carpeta *media* es muy pesada para Github, estos datos se tienen que descargar de la carpeta Drive que se les compartió para que el código pueda correr correctamente

## html

En caso de que usted no maneje python, se puede ver la presentación dándole click al archivo html que viene en el repositorio