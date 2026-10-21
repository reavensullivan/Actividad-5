# CHANGELOG

## v1.0.0 - 2026-10-20

### Datos
- Original Iris de scikit-learn: 150 registros.
- Estandarización y validación de tipos y medidas.
- Eliminación documentada de una fila idéntica: 149 registros.
- Conservación de posibles atípicos.
- División estratificada: 119 registros de entrenamiento y 30 de prueba.
- Registro de versión v1, huellas SHA-256 y semilla 42.

### Modelos
- Regresión logística y random forest con GridSearchCV de cinco particiones.
- Selección por F1 macro CV: regresion_logistica.
- Registro de parámetros, métricas, modelos y artefactos en MLflow.
- Evaluación final: ambos modelos acertaron 28 de 30 registros.
- McNemar sin pares discordantes; p=1 por convención.

### Correcciones
- Normalización de separadores de rutas para Windows.
- Uso explícito de cloudpickle ante el rechazo de numpy.dtype por skops.

### Reproducibilidad
- datos_prep.py creado y verificado contra los datos limpios.
- train.py creado y con sintaxis verificada.
- El script todavía no se ha ejecutado como proceso independiente.
- Resultados actuales obtenidos desde el notebook.
- Versiones registradas en requirements.txt y resultados/entorno.json.
