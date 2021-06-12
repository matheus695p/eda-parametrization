# tires optimizer
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# Descripción del funcionamiento

Post conexión vía API-URL firmada, llegaran los archivos al bucket de raw data, en el cúal habrán tópicos de SNS escuchando la llegada de los datos
de esta manera cada vez que llegué un archivo, se gatillarán las lambdas de limpieza de las distintas fuentes de datos.

# Descripción de stack

Estos serán los stacks de limpieza de cada una de las fuentes de datos, donde se estandarizarán nombres de columnas, formato de entrega de los datos, etc.
Constará de los siguientes stacks:
- gps/           --> donde estará la lambda function suscrita a tópico SNS en el bucket de raw en la ruta gps/, que hará la limpieza de gps
- loads/         --> donde estará la lambda function suscrita a tópico SNS en el bucket de raw en la ruta loads/, que hará la limpieza de loads
- mems/          --> donde estará la lambda function suscrita a tópico SNS en el bucket de raw en la ruta mems/, que hará la limpieza de mems
- maintenance/   --> donde estará la lambda function suscrita a tópico SNS en el bucket de raw en la ruta maintenance/, que hará la limpieza de maintenance

# ¿Cómo levantar el stack?

```sh
cd tires-optimizer/src/cleaning-process   --> moverse al directorio del template
bash deploy.sh                            --> archivo .sh que hará el levantamiento de los recursos en el la cuenta especificada vía aws-cli (command line interface), para hacer                                                 esto se necesita una terminal unix o un emulador de consola para windows

````
