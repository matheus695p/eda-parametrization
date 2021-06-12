# src/ads/daily-process

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)##

## Este es un proceso diario que va a resumir a nivel de equipo y a nivel de neumatico, caracteristicas que tuvieron durante el día
ADS: columnas que integra a nivel de llave de equipo

```zh
### llaves del dataframe
equipo: equipo en cuestión
fecha: fecha del día

### metricas de mineral
toneladas_transportadas: toneladas movidas durante el día
toneladas_por_viaje: toneladas promedio por viaje

### calidad de gps, para ver las perdidas que hay de gps
calidad_equipo: calidad de gps asociada a este equipo 

### metricas de tiempos en los diferentes estados
t_encendido: tiempo encendido según todas las fuentes de datos
t_encendido_gps: tiempo encendido según gps
t_encendido_loads: tiempo encendido según loads
t_encendido_mems: tiempo encendido según mems
t_subiendo_c: tiempo subiendo cargado en [hrs]
t_subiendo_v: tiempo subiendo vacio en [hrs]
t_bajando_c: tiempo bajando cargado en [hrs]
t_bajando_v: tiempo bajando vacio en [hrs]
t_plano_c: tiempo plano cargado en [hrs]
t_plano_v: tiempo plano vacio en [hrs]
### metricas de distancias recorridas en cada uno de los estados
dist_subiendo_c: distancia recorrida subiendo cargado en [km]
dist_subiendo_v: distancia recorrida subiendo vacio en [km]
dist_bajando_c: distancia recorrida bajando cargado en [km]
dist_bajando_v: distancia recorrida bajando vacio en [km]
dist_plano_c: distancia recorrida plano cargado en [km]
dist_plano_v: distancia recorrida plano vacio en [km]

### metricas de velocidades en cada uno de los estados
v_subiendo_c: velocidad subiendo cargado en [km / hrs] 
v_subiendo_v: velocidad subiendo vacio en [km / hrs]
v_bajando_c: velocidad bajando cargado en [km / hrs]
v_bajando_v: velocidad bajando vacio en [km / hrs]
v_plano_c: velocidad plano cargado en [km / hrs]
v_plano_v: velocidad plano vacio en [km / hrs]
vr_subiendo_c: velocidad subiendo cargado en [km / hrs] con velocidades mayores a 5 [km/hr]
vr_subiendo_v: velocidad subiendo vacio en [km / hrs] con velocidades mayores a 5 [km/hr]
vr_bajando_c: velocidad bajando cargado en [km / hrs] con velocidades mayores a 5 [km/hr]
vr_bajando_v: velocidad bajando vacio en [km / hrs] con velocidades mayores a 5 [km/hr]
vr_plano_c: velocidad plano cargado en [km / hrs] con velocidades mayores a 5 [km/hr]
vr_plano_v: velocidad plano vacio en [km / hrs] con velocidades mayores a 5 [km/hr]

### metricas de sumas de todas las metricas temporales que se realizaron
sum_angulo: sumatoria de los angulos
sum_angulo_positivo: sumatoria de los angulos positivos
sum_angulo_negativo: sumatoria de los angulos negativos
sum_aceleracion: sumatoria de la aceleración
sum_aceleracion_positiva: sumatoria de la aceleración positivas
sum_aceleracion_negativa: sumatoria de la aceleración negativas
sum_f_newton: sumatoria de fuerzas F = M * a
sum_f_newton_positiva: sumatoria de fuerzas F = M * a postivo
sum_f_newton_neagtiva: sumatoria de fuerzas F = M * a negativas
sum_f_motriz: sumatoria de fuerzas motriz del día 
sum_f_motriz_positiva: sumatoria de fuerzas motriz del día positiva
sum_f_motriz_negativa: sumatoria de fuerzas motriz del día negativa
sum_fn_total: sumatoria de fuerzas normales total equipo
sum_fn_delantera: sumatoria de fuerzas normales delantera
sum_fn_trasera: sumatoria de fuerzas normales trasera
sum_fr_delantera: sumatoria de fuerzas roce delantera
sum_fr_trasera: sumatoria de fuerzas roce trasera
sum_fn_neumatico_delantera: sumatoria de fuerzas normales delantera por neumático
sum_fn_neumatico_trasera: sumatoria de fuerzas normales trasera por neumático
sum_fr_neumatico_delantera: sumatoria de fuerzas de roce delantera por neumático
sum_fr_neumatico_trasera: sumatoria de fuerzas de roce trasera por neumático
sum_w_neto: sumatoria de trabajo neto realizado por el equipo
sum_w_neto_positivo: sumatoria de trabajo neto realizado por el equipo, positivo solamente
sum_w_neto_negativo: sumatoria de trabajo neto realizado por el equipo, negativo solamente
sum_wfr_delantera: sumatoria de trabajo neto realizado por el roce en la parte delantera
sum_wfr_trasera: sumatoria de trabajo neto realizado por el roce en la parte trasera
sum_wfr_neumatico_delantera: sumatoria de trabajo neto realizado por el roce en la parte delantera por neumático
sum_wfr_neumatico_trasera: sumatoria de trabajo neto realizado por el roce en la parte trasera por neumático

### metricas de promedio de todas las metricas temporales que se realizaron
mean_angulo: promedio de los angulos
mean_angulo_positivo: promedio de los angulos positivos
mean_angulo_negativo: promedio de los angulos negativos
mean_aceleracion: promedio de la aceleración
mean_aceleracion_positiva: promedio de la aceleración positivas
mean_aceleracion_negativa: promedio de la aceleración negativas
mean_f_newton: promedio de fuerzas F = M * a
mean_f_newton_positiva: promedio de fuerzas F = M * a postivo
mean_f_newton_neagtiva: promedio de fuerzas F = M * a negativas
mean_f_motriz: promedio de fuerzas motriz del día 
mean_f_motriz_positiva: promedio de fuerzas motriz del día positiva
mean_f_motriz_negativa: promedio de fuerzas motriz del día negativa
mean_fn_total: promedio de fuerzas normales total equipo
mean_fn_delantera: promedio de fuerzas normales delantera
mean_fn_trasera: promedio de fuerzas normales trasera
mean_fr_delantera: promedio de fuerzas roce delantera
mean_fr_trasera: promedio de fuerzas roce trasera
mean_fn_neumatico_delantera: promedio de fuerzas normales delantera por neumático
mean_fn_neumatico_trasera: promedio de fuerzas normales trasera por neumático
mean_fr_neumatico_delantera: promedio de fuerzas de roce delantera por neumático
mean_fr_neumatico_trasera: promedio de fuerzas de roce trasera por neumático
mean_w_neto: promedio de trabajo neto realizado por el equipo
mean_w_neto_positivo: promedio de trabajo neto realizado por el equipo, positivo solamente
mean_w_neto_negativo: promedio de trabajo neto realizado por el equipo, negativo solamente
mean_wfr_delantera: promedio de trabajo neto realizado por el roce en la parte delantera
mean_wfr_trasera: promedio de trabajo neto realizado por el roce en la parte trasera
mean_wfr_neumatico_delantera: promedio de trabajo neto realizado por el roce en la parte delantera por neumático
mean_wfr_neumatico_trasera: promedio de trabajo neto realizado por el roce en la parte trasera por neumático

### metricas de medianas de todas las metricas temporales que se realizaron
median_angulo: mediana de los angulos
median_angulo_positivo: mediana de los angulos positivos
median_angulo_negativo: mediana de los angulos negativos
median_aceleracion: mediana de la aceleración
median_aceleracion_positiva: mediana de la aceleración positivas
median_aceleracion_negativa: mediana de la aceleración negativas
median_f_newton: mediana de fuerzas F = M * a
median_f_newton_positiva: mediana de fuerzas F = M * a postivo
median_f_newton_neagtiva: mediana de fuerzas F = M * a negativas
median_f_motriz: mediana de fuerzas motriz del día 
median_f_motriz_positiva: mediana de fuerzas motriz del día positiva
median_f_motriz_negativa: mediana de fuerzas motriz del día negativa
median_fn_total: mediana de fuerzas normales total equipo
median_fn_delantera: mediana de fuerzas normales delantera
median_fn_trasera: mediana de fuerzas normales trasera
median_fr_delantera: mediana de fuerzas roce delantera
median_fr_trasera: mediana de fuerzas roce trasera
median_fn_neumatico_delantera: mediana de fuerzas normales delantera por neumático
median_fn_neumatico_trasera: mediana de fuerzas normales trasera por neumático
median_fr_neumatico_delantera: mediana de fuerzas de roce delantera por neumático
median_fr_neumatico_trasera: mediana de fuerzas de roce trasera por neumático
median_w_neto: mediana de trabajo neto realizado por el equipo
median_w_neto_positivo: mediana de trabajo neto realizado por el equipo, positivo solamente
median_w_neto_negativo: mediana de trabajo neto realizado por el equipo, negativo solamente
median_wfr_delantera: mediana de trabajo neto realizado por el roce en la parte delantera
median_wfr_trasera: mediana de trabajo neto realizado por el roce en la parte trasera
median_wfr_neumatico_delantera: mediana de trabajo neto realizado por el roce en la parte delantera por neumático
median_wfr_neumatico_trasera: mediana de trabajo neto realizado por el roce en la parte trasera por neumático

### metricas de conteo de registros en cada una de las bases de datos
conteo_gps: conteo de los registros de gps
conteo_loads: conteo de los registros de loads
conteo_mems: conteo de los registros de mems

### metricas de kurtosis de diferentes series de tiempo
kurt_angulo: kurtosis de los angulos
kurt_angulo_positivo: kurtosis de los angulos positivos
kurt_angulo_negativo: kurtosis de los angulos negativos
kurt_aceleracion: kurtosis de la aceleración
kurt_aceleracion_positiva: kurtosis de la aceleración positivas
kurt_aceleracion_negativa: kurtosis de la aceleración negativas
kurt_f_newton: kurtosis de fuerzas F = M * a
kurt_f_newton_positiva: kurtosis de fuerzas F = M * a postivo
kurt_f_newton_neagtiva: kurtosis de fuerzas F = M * a negativas
kurt_f_motriz: kurtosis de fuerzas motriz del día 
kurt_f_motriz_positiva: kurtosis de fuerzas motriz del día positiva
kurt_f_motriz_negativa: kurtosis de fuerzas motriz del día negativa
kurt_fn_total: kurtosis de fuerzas normales total equipo
kurt_fn_delantera: kurtosis de fuerzas normales delantera
kurt_fn_trasera: kurtosis de fuerzas normales trasera
kurt_fr_delantera: kurtosis de fuerzas roce delantera
kurt_fr_trasera: kurtosis de fuerzas roce trasera
kurt_fn_neumatico_delantera: kurtosis de fuerzas normales delantera por neumático
kurt_fn_neumatico_trasera: kurtosis de fuerzas normales trasera por neumático
kurt_fr_neumatico_delantera: kurtosis de fuerzas de roce delantera por neumático
kurt_fr_neumatico_trasera: kurtosis de fuerzas de roce trasera por neumático
kurt_w_neto: kurtosis de trabajo neto realizado por el equipo
kurt_w_neto_positivo: kurtosis de trabajo neto realizado por el equipo, positivo solamente
kurt_w_neto_negativo: kurtosis de trabajo neto realizado por el equipo, negativo solamente
kurt_wfr_delantera: kurtosis de trabajo neto realizado por el roce en la parte delantera
kurt_wfr_trasera: kurtosis de trabajo neto realizado por el roce en la parte trasera
kurt_wfr_neumatico_delantera: kurtosis de trabajo neto realizado por el roce en la parte delantera por neumático
kurt_wfr_neumatico_trasera: kurtosis de trabajo neto realizado por el roce en la parte trasera por neumático

### metricas autoregresion con diferentes lags = 1, 2, 3
#### lag= 1
ac_1_angulo: autocorrelacion lag 1 de los angulos
ac_1_angulo_positivo: autocorrelacion lag 1 de los angulos positivos
ac_1_angulo_negativo: autocorrelacion lag 1 de los angulos negativos
ac_1_aceleracion: autocorrelacion lag 1 de la aceleración
ac_1_aceleracion_positiva: autocorrelacion lag 1 de la aceleración positivas
ac_1_aceleracion_negativa: autocorrelacion lag 1 de la aceleración negativas
ac_1_f_newton: autocorrelacion lag 1 de fuerzas F = M * a
ac_1_f_newton_positiva: autocorrelacion lag 1 de fuerzas F = M * a postivo
ac_1_f_newton_neagtiva: autocorrelacion lag 1 de fuerzas F = M * a negativas
ac_1_f_motriz: autocorrelacion lag 1 de fuerzas motriz del día 
ac_1_f_motriz_positiva: autocorrelacion lag 1 de fuerzas motriz del día positiva
ac_1_f_motriz_negativa: autocorrelacion lag 1 de fuerzas motriz del día negativa
ac_1_fn_total: autocorrelacion lag 1 de fuerzas normales total equipo
ac_1_fn_delantera: autocorrelacion lag 1 de fuerzas normales delantera
ac_1_fn_trasera: autocorrelacion lag 1 de fuerzas normales trasera
ac_1_fr_delantera: autocorrelacion lag 1 de fuerzas roce delantera
ac_1_fr_trasera: autocorrelacion lag 1 de fuerzas roce trasera
ac_1_fn_neumatico_delantera: autocorrelacion lag 1 de fuerzas normales delantera por neumático
ac_1_fn_neumatico_trasera: autocorrelacion lag 1 de fuerzas normales trasera por neumático
ac_1_fr_neumatico_delantera: autocorrelacion lag 1 de fuerzas de roce delantera por neumático
ac_1_fr_neumatico_trasera: autocorrelacion lag 1 de fuerzas de roce trasera por neumático
ac_1_w_neto: autocorrelacion lag 1 de trabajo neto realizado por el equipo
ac_1_w_neto_positivo: autocorrelacion lag 1 de trabajo neto realizado por el equipo, positivo solamente
ac_1_w_neto_negativo: autocorrelacion lag 1 de trabajo neto realizado por el equipo, negativo solamente
ac_1_wfr_delantera: autocorrelacion lag 1 de trabajo neto realizado por el roce en la parte delantera
ac_1_wfr_trasera: autocorrelacion lag 1 de trabajo neto realizado por el roce en la parte trasera
ac_1_wfr_neumatico_delantera: autocorrelacion lag 1 de trabajo neto realizado por el roce en la parte delantera por neumático
ac_1_wfr_neumatico_trasera: autocorrelacion lag 1 de trabajo neto realizado por el roce en la parte trasera por neumático
#### lag= 2
ac_2_angulo: autocorrelacion lag 2 de los angulos
ac_2_angulo_positivo: autocorrelacion lag 2 de los angulos positivos
ac_2_angulo_negativo: autocorrelacion lag 2 de los angulos negativos
ac_2_aceleracion: autocorrelacion lag 2 de la aceleración
ac_2_aceleracion_positiva: autocorrelacion lag 2 de la aceleración positivas
ac_2_aceleracion_negativa: autocorrelacion lag 2 de la aceleración negativas
ac_2_f_newton: autocorrelacion lag 2 de fuerzas F = M * a
ac_2_f_newton_positiva: autocorrelacion lag 2 de fuerzas F = M * a postivo
ac_2_f_newton_neagtiva: autocorrelacion lag 2 de fuerzas F = M * a negativas
ac_2_f_motriz: autocorrelacion lag 2 de fuerzas motriz del día 
ac_2_f_motriz_positiva: autocorrelacion lag 2 de fuerzas motriz del día positiva
ac_2_f_motriz_negativa: autocorrelacion lag 2 de fuerzas motriz del día negativa
ac_2_fn_total: autocorrelacion lag 2 de fuerzas normales total equipo
ac_2_fn_delantera: autocorrelacion lag 2 de fuerzas normales delantera
ac_2_fn_trasera: autocorrelacion lag 2 de fuerzas normales trasera
ac_2_fr_delantera: autocorrelacion lag 2 de fuerzas roce delantera
ac_2_fr_trasera: autocorrelacion lag 2 de fuerzas roce trasera
ac_2_fn_neumatico_delantera: autocorrelacion lag 2 de fuerzas normales delantera por neumático
ac_2_fn_neumatico_trasera: autocorrelacion lag 2 de fuerzas normales trasera por neumático
ac_2_fr_neumatico_delantera: autocorrelacion lag 2 de fuerzas de roce delantera por neumático
ac_2_fr_neumatico_trasera: autocorrelacion lag 2 de fuerzas de roce trasera por neumático
ac_2_w_neto: autocorrelacion lag 2 de trabajo neto realizado por el equipo
ac_2_w_neto_positivo: autocorrelacion lag 2 de trabajo neto realizado por el equipo, positivo solamente
ac_2_w_neto_negativo: autocorrelacion lag 2 de trabajo neto realizado por el equipo, negativo solamente
ac_2_wfr_delantera: autocorrelacion lag 2 de trabajo neto realizado por el roce en la parte delantera
ac_2_wfr_trasera: autocorrelacion lag 2 de trabajo neto realizado por el roce en la parte trasera
ac_2_wfr_neumatico_delantera: autocorrelacion lag 2 de trabajo neto realizado por el roce en la parte delantera por neumático
ac_2_wfr_neumatico_trasera: autocorrelacion lag 2 de trabajo neto realizado por el roce en la parte trasera por neumático
#### lag= 3
ac_3_angulo: autocorrelacion lag 3 de los angulos
ac_3_angulo_positivo: autocorrelacion lag 3 de los angulos positivos
ac_3_angulo_negativo: autocorrelacion lag 3 de los angulos negativos
ac_3_aceleracion: autocorrelacion lag 3 de la aceleración
ac_3_aceleracion_positiva: autocorrelacion lag 3 de la aceleración positivas
ac_3_aceleracion_negativa: autocorrelacion lag 3 de la aceleración negativas
ac_3_f_newton: autocorrelacion lag 3 de fuerzas F = M * a
ac_3_f_newton_positiva: autocorrelacion lag 3 de fuerzas F = M * a postivo
ac_3_f_newton_neagtiva: autocorrelacion lag 3 de fuerzas F = M * a negativas
ac_3_f_motriz: autocorrelacion lag 3 de fuerzas motriz del día 
ac_3_f_motriz_positiva: autocorrelacion lag 3 de fuerzas motriz del día positiva
ac_3_f_motriz_negativa: autocorrelacion lag 3 de fuerzas motriz del día negativa
ac_3_fn_total: autocorrelacion lag 3 de fuerzas normales total equipo
ac_3_fn_delantera: autocorrelacion lag 3 de fuerzas normales delantera
ac_3_fn_trasera: autocorrelacion lag 3 de fuerzas normales trasera
ac_3_fr_delantera: autocorrelacion lag 3 de fuerzas roce delantera
ac_3_fr_trasera: autocorrelacion lag 3 de fuerzas roce trasera
ac_3_fn_neumatico_delantera: autocorrelacion lag 3 de fuerzas normales delantera por neumático
ac_3_fn_neumatico_trasera: autocorrelacion lag 3 de fuerzas normales trasera por neumático
ac_3_fr_neumatico_delantera: autocorrelacion lag 3 de fuerzas de roce delantera por neumático
ac_3_fr_neumatico_trasera: autocorrelacion lag 3 de fuerzas de roce trasera por neumático
ac_3_w_neto: autocorrelacion lag 3 de trabajo neto realizado por el equipo
ac_3_w_neto_positivo: autocorrelacion lag 3 de trabajo neto realizado por el equipo, positivo solamente
ac_3_w_neto_negativo: autocorrelacion lag 3 de trabajo neto realizado por el equipo, negativo solamente
ac_3_wfr_delantera: autocorrelacion lag 3 de trabajo neto realizado por el roce en la parte delantera
ac_3_wfr_trasera: autocorrelacion lag 3 de trabajo neto realizado por el roce en la parte trasera
ac_3_wfr_neumatico_delantera: autocorrelacion lag 3 de trabajo neto realizado por el roce en la parte delantera por neumático
ac_3_wfr_neumatico_trasera: autocorrelacion lag 3 de trabajo neto realizado por el roce en la parte trasera por neumático
```

