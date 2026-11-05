# Predicción de precios de casas en California

Proyecto de regresión con scikit-learn: predecir el valor de las casas de una zona de California a partir de datos del censo. Comparo un árbol de decisión sin límite, un árbol con el tamaño ajustado y un Random Forest.

## El problema

Quiero predecir un **número**: el valor mediano de las casas en una zona. Como el resultado es un número y no una categoría, es un problema de **regresión**. Para medir qué tan bien predice el modelo uso el **MAE** (error absoluto medio), que dice cuánto se equivoca el modelo en promedio.

## Los datos

- **Fuente:** dataset *California Housing*, incluido en scikit-learn (`fetch_california_housing`). Viene del censo de Estados Unidos de 1990.
- **Tamaño:** 20,640 filas y 9 columnas. Cada fila es una **zona** del censo, no una casa individual.
- **Columnas usadas para predecir (8):**

| Columna | Qué significa |
|---|---|
| `MedInc` | Ingreso mediano de la zona (en decenas de miles de dólares) |
| `HouseAge` | Edad mediana de las casas |
| `AveRooms` | Promedio de cuartos por casa |
| `AveBedrms` | Promedio de recámaras por casa |
| `Population` | Número de habitantes de la zona |
| `AveOccup` | Promedio de personas por casa |
| `Latitude` | Latitud |
| `Longitude` | Longitud |

- **Lo que se predice:** `MedHouseVal`, el valor mediano de las casas, en unidades de 100,000 dólares. Un valor de 2.0 equivale a unos 200,000 dólares de 1990.
- **Valores faltantes:** ninguno.

## Lo que hice

1. Exploré los datos con `.head()`, `.shape`, `.describe()` y `.isnull().sum()`.
2. Separé las columnas para predecir (`X`) de la respuesta (`y`) y dividí los datos en entrenamiento (15,480 zonas) y validación (5,160 zonas) con `train_test_split` y `random_state=1`.
3. Entrené un árbol de decisión sin límite y medí su error con datos de entrenamiento y de validación.
4. Probé distintos valores de `max_leaf_nodes` para encontrar el mejor tamaño de árbol.
5. Entrené un Random Forest y comparé los tres modelos.
6. Revisé qué columnas usó más el Random Forest (`feature_importances_`).

## Resultados

### Comparación de modelos

| Modelo | MAE en validación | Error aproximado |
|---|---|---|
| Árbol sin límite | 0.464 | 46,400 dólares |
| Árbol con 500 hojas | 0.412 | 41,200 dólares |
| **Random Forest** | **0.331** | **33,100 dólares** |

El Random Forest redujo el error en casi **29%** respecto al árbol sin límite (de 0.464 a 0.331). Limitar el árbol a 500 hojas lo redujo en un 11%.

### Overfitting en el árbol sin límite

El árbol sin límite tuvo un error de **0.0 en los datos de entrenamiento** y de **0.464 en los de validación**. Esto es overfitting: el árbol memorizó las zonas de entrenamiento, pero esas reglas tan específicas no sirven para zonas nuevas.

### Búsqueda del mejor tamaño de árbol

| max_leaf_nodes | MAE en validación |
|---|---|
| 5 | 0.632 |
| 25 | 0.518 |
| 50 | 0.472 |
| 100 | 0.442 |
| 250 | 0.413 |
| **500** | **0.412** |
| 1000 | 0.422 |
| 2000 | 0.436 |

El error baja hasta las 500 hojas (con pocas hojas el árbol no distingue bien las zonas, es decir, underfitting) y vuelve a subir después (con demasiadas hojas empieza a memorizar, es decir, overfitting). 250 hojas da un resultado casi igual.

### Qué columnas pesaron más (Random Forest)

| Columna | Importancia |
|---|---|
| `MedInc` | 0.522 |
| `AveOccup` | 0.141 |
| `Latitude` | 0.092 |
| `Longitude` | 0.089 |
| `HouseAge` | 0.053 |
| `AveRooms` | 0.043 |
| `AveBedrms` | 0.031 |
| `Population` | 0.030 |

El ingreso mediano (`MedInc`) fue la columna más usada, con más de la mitad de la importancia total. La ubicación (latitud y longitud juntas) suma 0.181.

## Qué aprendí

- **Overfitting:** un modelo puede parecer perfecto si solo se mide con los datos con los que entrenó. Por eso hay que evaluarlo con datos de validación que nunca vio.
- **Ajustar la complejidad:** al probar varios tamaños de árbol vi la curva en forma de "U": el error baja, toca un mínimo y vuelve a subir.
- **Muchos árboles superan a uno solo:** el Random Forest dio el menor error sin necesidad de ajustar parámetros.
- **Cuidar los nombres de las variables:** en un momento sobrescribí por accidente la variable de un modelo con la de otro, y la tabla de resultados salió con dos números idénticos. Aprendí a desconfiar cuando dos resultados salen exactamente iguales.

## Limitaciones

- Los precios son de **1990**, así que el modelo no sirve para estimar precios actuales.
- Los valores más altos de `MedHouseVal` están **cortados en 5.0** (500,000 dólares), por lo que las zonas más caras aparecen todas con el mismo valor.
- Cada fila es una zona del censo, no una casa individual, y las columnas son promedios y medianas.
- La importancia de las columnas indica cuáles usó más el modelo, **no** que causen el precio.
- Medí el error con una sola división de entrenamiento y validación, y el Random Forest usa sus parámetros por defecto.

## Cómo mejorarlo

- Usar **validación cruzada** para tener una medida más estable del error.
- Ajustar parámetros del Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`).
- Probar otros modelos, como Gradient Boosting.
- Crear columnas nuevas (por ejemplo, cuartos por persona).
- Graficar las predicciones contra los valores reales para ver dónde falla el modelo.

## Cómo correr el proyecto

El notebook corre en Google Colab sin instalar nada. Para correrlo en tu computadora:

```bash
pip install pandas scikit-learn jupyter
jupyter notebook
```

Abre el notebook y corre las celdas en orden. El dataset se descarga automáticamente la primera vez.
