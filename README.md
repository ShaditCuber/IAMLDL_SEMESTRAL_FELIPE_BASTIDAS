# Proyecto de Curso de Machine Learning

Este proyecto es parte de un curso de Machine Learning y presenta un análisis comparativo de diferentes modelos entrenados en un conjunto de datos específico. El objetivo es determinar el rendimiento de cada modelo midiendo su precisión (accuracy).

## Instrucciones de Instalación y Ejecución

Siga estas instrucciones para obtener una copia del proyecto en funcionamiento en su máquina local.

### Prerrequisitos

Debes tener `git` y `docker` instalados en tu sistema para clonar el repositorio y ejecutar el proyecto en un contenedor de Docker.

- [Instrucciones para instalar Git](https://git-scm.com/downloads)
- [Instrucciones para instalar Docker](https://docs.docker.com/get-docker/)

### Clonar el Repositorio

Para clonar el repositorio, ejecuta el siguiente comando:

```bash
git clone https://github.com/ShaditCuber/IAMLDL_SEMESTRAL_FELIPE_BASTIDAS
cd IAMLDL_SEMESTRAL_FELIPE_BASTIDAS
```

### Construir Imagen Docker

```bash
docker build -t iamldl_felipe_bastidas .
```


### Ejecutar el Contenedor

* Se utiliza -d para dejar el proceso en segundo plano
* Se utiliza -it para poder ingresar al docker y que no se cierre apenas inicie

```bash
docker run -dit iamldl_felipe_bastidas
```


### Ingresar al Contenedor

Abrir una consola nueva y ejecutar el siguiente comando

```bash
docker attach *ingrese hash del contenedor ejecutandose*
```


### Ejecutar el Script de Entrenamiento

Dentro de la misma consola que abra el contenedor se ejecuta lo siguiente

```bash
python3 index.py
```

### Visualización de Resultados

Puedes visualizar los resultados de la precisión de los modelos a través de los archivos de las matrices de confusión generados, que se encuentran en el directorio matrices_de_confusion.


### Autores

* Felipe Bastidas - [Shadit Cuber](https://github.com/ShaditCuber)


---

⌨️ con ❤️ por [Felipe Bastidas](https://github.com/ShaditCuber)