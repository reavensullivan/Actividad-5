"""Entrenamiento reproducible de Iris y registro local en MLflow.

Ejecutar desde la raíz del proyecto:
    python fuentes/train.py
"""

from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn

from scipy.stats import binomtest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature

from datos_prep import cargar_y_limpiar, VARIABLES, CLASES, SEMILLA


BASE = Path(__file__).resolve().parent.parent


def guardar_json(ruta, contenido):
    ruta.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    for carpeta in [
        "datos/datos_ini",
        "datos/datos_limp",
        "resultados",
        "modelos",
        "mlartifacts",
    ]:
        (BASE / carpeta).mkdir(parents=True, exist_ok=True)

    # Usar el original existente; obtenerlo si falta.
    ruta_original = BASE / "datos/datos_ini/iris_original.csv"

    if not ruta_original.exists():
        iris = load_iris(as_frame=True)
        iris.frame.to_csv(ruta_original, index=False)
        (ruta_original.parent / "descripcion_iris.txt").write_text(
            iris.DESCR, encoding="utf-8"
        )

    datos, resumen = cargar_y_limpiar(ruta_original)

    ruta_limpia = BASE / "datos/datos_limp/iris_limpio_v1.csv"
    datos.to_csv(ruta_limpia, index=False)
    guardar_json(BASE / "resultados/resumen_limpieza.json", resumen)

    X = datos[VARIABLES]
    y = datos["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEMILLA,
        stratify=y,
    )

    ruta_train = BASE / "datos/datos_limp/entrenamiento_v1.csv"
    ruta_test = BASE / "datos/datos_limp/prueba_v1.csv"

    X_train.join(y_train).to_csv(ruta_train, index_label="fila_id")
    X_test.join(y_test).to_csv(ruta_test, index_label="fila_id")

    version = {
        "version": "v1",
        "semilla": SEMILLA,
        "proporcion_prueba": 0.20,
        "division": "estratificada por especie",
        "filas_entrenamiento": len(X_train),
        "filas_prueba": len(X_test),
        "sha256": {
            ruta.relative_to(BASE).as_posix(): hashlib.sha256(
                ruta.read_bytes()
            ).hexdigest()
            for ruta in [
                ruta_original, ruta_limpia, ruta_train, ruta_test
            ]
        },
    }

    ruta_version = BASE / "datos/version_datos_v1.json"
    guardar_json(ruta_version, version)

    # Configuración de MLflow independiente del directorio de ejecución.
    tracking_uri = "sqlite:///" + (BASE / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(tracking_uri)

    cliente = mlflow.tracking.MlflowClient()
    experimento = "Actividad5_Iris"

    if cliente.get_experiment_by_name(experimento) is None:
        cliente.create_experiment(
            experimento,
            artifact_location=(BASE / "mlartifacts").as_uri(),
        )

    mlflow.set_experiment(experimento)

    cv = list(
        StratifiedKFold(
            n_splits=5, shuffle=True, random_state=SEMILLA
        ).split(X_train, y_train)
    )

    configuraciones = {
        "regresion_logistica": (
            Pipeline([
                ("imputacion", SimpleImputer(strategy="median")),
                ("escalado", StandardScaler()),
                ("modelo", LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=SEMILLA,
                )),
            ]),
            {"modelo__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "random_forest": (
            Pipeline([
                ("imputacion", SimpleImputer(strategy="median")),
                ("modelo", RandomForestClassifier(
                    random_state=SEMILLA, n_jobs=1
                )),
            ]),
            {
                "modelo__n_estimators": [100, 200],
                "modelo__max_depth": [None, 3, 5],
                "modelo__min_samples_leaf": [1, 2],
            },
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "roc_auc_ovr_macro": "roc_auc_ovr",
    }

    resumen_cv = []
    modelos = {}
    runs = {}

    # Primero seleccionar usando solo entrenamiento y validación.
    for nombre, (pipeline, rejilla) in configuraciones.items():
        print(f"Entrenando: {nombre}")

        with mlflow.start_run(run_name=nombre) as run:
            inicio = time.perf_counter()

            busqueda = GridSearchCV(
                pipeline,
                rejilla,
                scoring=scoring,
                refit="f1_macro",
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
                error_score="raise",
            )
            busqueda.fit(X_train, y_train)

            segundos = time.perf_counter() - inicio
            modelo = busqueda.best_estimator_
            mejor = busqueda.best_index_

            tabla = pd.DataFrame(busqueda.cv_results_)
            ruta_cv = BASE / f"resultados/gridsearch_{nombre}.csv"
            tabla.to_csv(ruta_cv, index=False)

            joblib.dump(modelo, BASE / f"modelos/{nombre}.joblib")

            mlflow.set_tags({
                "tipo": "entrenamiento",
                "dataset": "Iris",
                "version_datos": "v1",
                "criterio_seleccion": "f1_macro_cv",
            })
            mlflow.log_params({
                "algoritmo": nombre,
                "semilla": SEMILLA,
                "cv_particiones": 5,
                "filas_entrenamiento": len(X_train),
                "numero_combinaciones": len(tabla),
                "datos_sha256": version["sha256"][
                    "datos/datos_limp/iris_limpio_v1.csv"
                ],
                "serializacion": "cloudpickle",
            })
            mlflow.log_params(busqueda.best_params_)
            mlflow.log_dict(rejilla, "configuracion/rejilla.json")

            mlflow.log_dict({
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "mlflow": mlflow.__version__,
            }, "configuracion/entorno.json")

            mlflow.log_dict({
                "particiones": [
                    {
                        "fold": numero,
                        "indices_entrenamiento": train.tolist(),
                        "indices_validacion": val.tolist(),
                    }
                    for numero, (train, val) in enumerate(cv, start=1)
                ],
                "nota": "Índices posicionales relativos a X_train.",
            }, "configuracion/particiones_cv.json")

            metricas_cv = {"busqueda_segundos": segundos}

            for metrica in scoring:
                for estadistico, prefijo in [
                    ("media", "mean"), ("std", "std")
                ]:
                    metricas_cv[f"cv_{metrica}_{estadistico}"] = float(
                        busqueda.cv_results_[
                            f"{prefijo}_test_{metrica}"
                        ][mejor]
                    )

            mlflow.log_metrics(metricas_cv)
            mlflow.log_artifact(str(ruta_cv), "validacion")
            mlflow.log_artifact(str(ruta_version), "datos")

            for archivo in ["datos_prep.py", "train.py"]:
                mlflow.log_artifact(
                    str(BASE / "fuentes" / archivo), "codigo"
                )

            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="modelo",
                signature=infer_signature(
                    X_train, modelo.predict(X_train)
                ),
                input_example=X_train.head(3),
                serialization_format="cloudpickle",
            )

            modelos[nombre] = modelo
            runs[nombre] = run.info.run_id

            resumen_cv.append({
                "modelo": nombre,
                "f1_macro_cv": metricas_cv["cv_f1_macro_media"],
                "std_f1_cv": metricas_cv["cv_f1_macro_std"],
                "accuracy_cv": metricas_cv["cv_accuracy_media"],
                "roc_auc_cv": metricas_cv[
                    "cv_roc_auc_ovr_macro_media"
                ],
                "tiempo_busqueda_s": segundos,
                "mejores_parametros": json.dumps(
                    busqueda.best_params_
                ),
                "run_id": run.info.run_id,
            })

    comparacion_cv = pd.DataFrame(resumen_cv)
    comparacion_cv["orden"] = comparacion_cv["modelo"].map({
        "regresion_logistica": 0,
        "random_forest": 1,
    })
    comparacion_cv = (
        comparacion_cv.sort_values(
            ["f1_macro_cv", "orden"], ascending=[False, True]
        ).drop(columns="orden").reset_index(drop=True)
    )

    seleccionado = comparacion_cv.iloc[0]["modelo"]
    comparacion_cv.to_csv(
        BASE / "resultados/comparacion_validacion_cruzada.csv",
        index=False,
    )
    guardar_json(BASE / "resultados/runs_modelos.json", runs)
    guardar_json(BASE / "resultados/seleccion_modelo.json", {
        "modelo": seleccionado,
        "criterio": "Mayor F1 macro CV; empate: regresión logística.",
    })

    # Evaluación final; no cambia la selección anterior.
    resultados = []
    predicciones = {}

    for nombre, modelo in modelos.items():
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)
        predicciones[nombre] = pred

        metricas = {
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision_macro": precision_score(
                y_test, pred, average="macro", zero_division=0
            ),
            "test_recall_macro": recall_score(
                y_test, pred, average="macro", zero_division=0
            ),
            "test_f1_macro": f1_score(
                y_test, pred, average="macro", zero_division=0
            ),
            "test_roc_auc_ovr_macro": roc_auc_score(
                y_test, proba,
                labels=modelo.classes_,
                multi_class="ovr",
                average="macro",
            ),
        }

        reporte = classification_report(
            y_test, pred,
            labels=modelo.classes_,
            target_names=[CLASES[int(c)] for c in modelo.classes_],
            output_dict=True,
            zero_division=0,
        )
        ruta_reporte = (
            BASE / f"resultados/reporte_clasificacion_{nombre}.csv"
        )
        pd.DataFrame(reporte).transpose().to_csv(ruta_reporte)

        detalle = X_test.copy()
        detalle["target_real"] = y_test
        detalle["target_predicho"] = pred

        for posicion, clase in enumerate(modelo.classes_):
            detalle[f"probabilidad_{CLASES[int(clase)]}"] = (
                proba[:, posicion]
            )

        ruta_pred = BASE / f"resultados/predicciones_{nombre}.csv"
        detalle.to_csv(ruta_pred, index_label="fila_id")

        with mlflow.start_run(run_id=runs[nombre]):
            mlflow.log_metrics(metricas)
            mlflow.set_tag(
                "seleccionado_por_cv",
                str(nombre == seleccionado).lower(),
            )
            mlflow.log_artifact(str(ruta_reporte), "evaluacion_prueba")
            mlflow.log_artifact(str(ruta_pred), "evaluacion_prueba")

        resultados.append({"modelo": nombre, **metricas})

    comparacion = pd.DataFrame(resultados)
    ruta_comparacion = BASE / "resultados/comparacion_prueba.csv"
    comparacion.to_csv(ruta_comparacion, index=False)

    # McNemar exacto sobre aciertos pareados.
    reales = y_test.to_numpy()
    lr = predicciones["regresion_logistica"] == reales
    rf = predicciones["random_forest"] == reales

    a = int(np.sum(lr & rf))
    b = int(np.sum(lr & ~rf))
    c = int(np.sum(~lr & rf))
    d = int(np.sum(~lr & ~rf))

    p_valor = (
        float(binomtest(b, b + c, p=0.5).pvalue)
        if b + c else 1.0
    )

    if b + c == 0:
        interpretacion = (
            "Sin pares discordantes; p=1 por convención. "
            "No demuestra equivalencia general."
        )
    elif p_valor < 0.05:
        interpretacion = "Se detecta diferencia en tasas de error."
    else:
        interpretacion = (
            "No se detecta diferencia en tasas de error. "
            "No demuestra equivalencia."
        )

    ruta_estadistica = BASE / "resultados/comparacion_estadistica.json"
    guardar_json(ruta_estadistica, {
        "prueba": "McNemar exacto bilateral",
        "ambos_aciertan": a,
        "solo_logistica_acierta": b,
        "solo_random_forest_acierta": c,
        "ambos_fallan": d,
        "pares_discordantes": b + c,
        "p_valor": p_valor,
        "nivel_significancia": 0.05,
        "interpretacion": interpretacion,
        "limitacion": (
            "Prueba pequeña; compara tasas de error, no F1 ni ROC-AUC."
        ),
    })

    ruta_pareada = BASE / "resultados/tabla_mcnemar.csv"
    pd.DataFrame(
        [[a, b], [c, d]],
        index=["Logística acierta", "Logística falla"],
        columns=["Random forest acierta", "Random forest falla"],
    ).to_csv(ruta_pareada)

    for run_id in runs.values():
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("mcnemar_p_valor", p_valor)
            for ruta in [
                ruta_comparacion, ruta_estadistica, ruta_pareada
            ]:
                mlflow.log_artifact(str(ruta), "comparacion")

    print(comparacion.to_string(index=False))
    print("\nSeleccionado por CV:", seleccionado)
    print("McNemar:", interpretacion)
    print("Registro en MLflow completado.")


if __name__ == "__main__":
    main()
