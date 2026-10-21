"""Preparación reproducible del dataset Iris."""

from pathlib import Path
import pandas as pd

SEMILLA = 42

VARIABLES = [
    "largo_sepalo_cm",
    "ancho_sepalo_cm",
    "largo_petalo_cm",
    "ancho_petalo_cm",
]

NOMBRES_COLUMNAS = {
    "sepal length (cm)": "largo_sepalo_cm",
    "sepal width (cm)": "ancho_sepalo_cm",
    "petal length (cm)": "largo_petalo_cm",
    "petal width (cm)": "ancho_petalo_cm",
}

CLASES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def limpiar_datos(datos):
    """Estandariza nombres y tipos, valida y elimina duplicados.

    Los nulos en características se conservan para imputarlos
    dentro del pipeline. Un objetivo nulo o inválido se rechaza.
    No elimina valores atípicos por criterio estadístico.
    """
    limpios = datos.copy().rename(columns=NOMBRES_COLUMNAS)

    requeridas = VARIABLES + ["target"]
    faltantes = set(requeridas) - set(limpios.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    limpios = limpios[requeridas].copy()

    for columna in requeridas:
        limpios[columna] = pd.to_numeric(
            limpios[columna], errors="raise"
        )

    for columna in VARIABLES:
        valores = limpios[columna].dropna()

        if not (
            valores.gt(0) & valores.lt(float("inf"))
        ).all():
            raise ValueError(
                f"Medidas no positivas o no finitas en {columna}"
            )

    if not limpios["target"].isin(CLASES).all():
        raise ValueError("Target contiene nulos o clases inválidas.")

    duplicados = int(limpios.duplicated().sum())
    nulos = int(limpios.isna().sum().sum())

    limpios = limpios.drop_duplicates().reset_index(drop=True)
    limpios["target"] = limpios["target"].astype("int64")

    resumen = {
        "version": "v1",
        "filas_originales": len(datos),
        "filas_limpias": len(limpios),
        "valores_nulos": nulos,
        "duplicados_eliminados": duplicados,
        "criterio_duplicados": (
            "Eliminar filas idénticas en características y "
            "objetivo antes de dividir. Sin identificadores "
            "no se puede verificar si eran flores distintas."
        ),
        "tratamiento_nulos": (
            "Imputación de características dentro del pipeline."
        ),
        "escalado": (
            "Se ajusta dentro del pipeline, solo con entrenamiento."
        ),
        "atipicos": "Se conservan; IQR no demuestra error.",
    }

    return limpios, resumen


def cargar_y_limpiar(ruta_csv):
    """Lee el CSV original y devuelve datos limpios y resumen."""
    originales = pd.read_csv(Path(ruta_csv))
    return limpiar_datos(originales)
