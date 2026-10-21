# Actividad 5: clasificación de especies Iris con MLflow

## Objetivo
Preparar y versionar datos públicos, comparar dos algoritmos,
ajustar hiperparámetros y registrar experimentos reproducibles en MLflow.

## Dataset y fuente
Se utiliza la copia de Iris distribuida con scikit-learn,
obtenida mediante load_iris(as_frame=True).

- Fuente pública: https://archive.ics.uci.edu/dataset/53/iris
- Documentación: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- Original: 150 registros y 5 columnas.
- Limpio: 149 registros, tras eliminar una fila idéntica.
- Clases: 0 = setosa, 1 = versicolor, 2 = virginica.
- No se presupone identidad con otras distribuciones históricas de Iris.

## Diccionario de datos

| Variable | Descripción | Tipo | Unidad |
|---|---|---|---|
| largo_sepalo_cm | Longitud del sépalo | Numérico | cm |
| ancho_sepalo_cm | Anchura del sépalo | Numérico | cm |
| largo_petalo_cm | Longitud del pétalo | Numérico | cm |
| ancho_petalo_cm | Anchura del pétalo | Numérico | cm |
| target | Especie codificada: 0, 1 o 2 | Entero categórico | No aplica |

fila_id representa la posición en los datos limpios, no el identificador
original de una flor. No se utiliza como predictor.

## Limpieza y supuestos
- Estandarización de nombres y validación de tipos numéricos.
- Validación de medidas positivas y clases permitidas.
- No se encontraron valores nulos.
- Eliminación de una fila idéntica antes de dividir los datos.
- Sin identificadores originales no se puede determinar si esa fila
  correspondía a una flor distinta; se documenta esta limitación.
- Conservación de posibles atípicos: IQR no demuestra errores.
- Las 11 detecciones por variable y especie no necesariamente
  representan 11 flores distintas.
- No hay características categóricas que requieran codificación.
- Imputación preventiva por mediana dentro de los pipelines.
- Escalado únicamente para regresión logística, dentro del pipeline.
- Se asume independencia entre registros y etiquetas correctas.
- El análisis exploratorio se realizó sobre los datos limpios completos.
  Las métricas de prueba no se usaron para ajustar hiperparámetros.

## Estructura
- datos/datos_ini/: originales y descripción.
- datos/datos_limp/: versión limpia y particiones.
- datos/version_datos_v1.json: huellas SHA-256 y configuración.
- fuentes/entrena.ipynb: notebook de análisis y entrenamiento.
- fuentes/datos_prep.py: funciones de limpieza.
- fuentes/train.py: entrenamiento, evaluación y registro.
- resultados/: tablas, gráficas, predicciones y análisis estadístico.
- modelos/: pipelines en formato joblib.
- evidencias/: capturas del panel MLflow.
- mlflow.db: base de datos de seguimiento local.
- mlartifacts/: modelos y artefactos de MLflow.
- requirements.txt: versiones de dependencias principales.
- CHANGELOG.md: cambios y decisiones.

## Entrenamiento y selección
División estratificada 80/20 con semilla 42:
119 registros de entrenamiento y 30 de prueba.

GridSearchCV utiliza las mismas cinco particiones estratificadas
para ambos algoritmos. El criterio de selección es F1 macro medio.
En empate exacto se prefiere regresión logística por menor complejidad.

Regresión logística:
- C: 0.01, 0.1, 1, 10, 100.
- Cinco combinaciones.
- C pequeño implica mayor regularización.

Random forest:
- n_estimators: 100, 200.
- max_depth: None, 3, 5.
- min_samples_leaf: 1, 2.
- Doce combinaciones.
- Se varían tamaño del bosque y complejidad de los árboles.

Precision, recall y F1 usan promedio macro.
ROC-AUC utiliza probabilidades, one-vs-rest y promedio macro.

## Resultados

| Modelo | F1 macro CV | DE CV | Accuracy prueba | Precision macro | Recall macro | F1 macro prueba | ROC-AUC prueba |
|---|---:|---:|---:|---:|---:|---:|---:|
| regresion_logistica | 0.9629 | 0.0556 | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9967 |
| random_forest | 0.9347 | 0.0743 | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9867 |

Modelo seleccionado por validación cruzada: **regresion_logistica**.

## Comparación estadística
Los dos modelos acertaron las mismas 28 flores y fallaron en las mismas 2.
No hubo pares discordantes en McNemar: se reporta p=1 por convención.
Esto no demuestra equivalencia general.
El contraste compara tasas de error, no F1 ni ROC-AUC.
La muestra de prueba de 30 registros limita la fuerza de las conclusiones.

## Reproducción
Usar preferentemente la versión de Python registrada en
resultados/entorno.json y un entorno virtual independiente.

Desde la raíz del proyecto, ejecutar:

    python -m pip install -r requirements.txt
    python fuentes/train.py

El script genera nuevas ejecuciones en MLflow y actualiza resultados
numéricos y modelos. Las gráficas se reproducen desde el notebook.

Para usar Jupyter, instalarlo en el mismo entorno y abrir
fuentes/entrena.ipynb. Ejecutar las celdas en orden.
BASE debe apuntar a la raíz del proyecto, aunque el notebook se abra
desde fuentes. Verificar esa ruta antes de ejecutar la preparación.

Las dependencias directas están fijadas; las transitivas todavía
no están completamente bloqueadas.

## Panel de MLflow
Desde la raíz del proyecto, ejecutar:

    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Abrir http://127.0.0.1:5000 y consultar Actividad5_Iris.
Si el panel ya está abierto, no iniciar otra instancia.

La base de datos registra rutas absolutas de artefactos locales.
Si el proyecto cambia de ubicación o computadora, ejecutar train.py
para generar nuevas ejecuciones con las rutas correspondientes.

Los modelos se guardan con joblib y cloudpickle.
Cargar únicamente archivos de confianza.

## Trazabilidad
Los archivos v1 se identifican mediante SHA-256.
Las huellas verifican contenido; no sustituyen el historial de Git.
Los resultados de todas las combinaciones están en gridsearch_*.csv.
Los identificadores de runs están en resultados/runs_modelos.json.

train.py incluye registro de código, entorno y particiones CV.
Estas incorporaciones corresponden a nuevas ejecuciones del script;
no se atribuyen a ejecuciones anteriores del notebook.

## Deuda técnica y MLOps (Importante por ser ejercicio acádemico)
- Ampliar datos y realizar validación externa.
- Considerar el optimismo de las métricas CV utilizadas para ajustar
  hiperparámetros; conservar una evaluación final independiente.
- Bloquear dependencias transitivas o crear un contenedor.
- Incorporar CI con validaciones de esquema, datos y ejecución.
- Migrar MLflow y artefactos a almacenamiento compartido para equipos.
- Incorporar commits y etiquetas de versión con Git.
- Si aumentan los datos, utilizar una herramienta de versionamiento de datos.
- Definir monitoreo de deriva, desempeño y calidad de datos.
- Establecer criterios de reentrenamiento y reversión.
- El proyecto es un prototipo académico sin despliegue en producción.
