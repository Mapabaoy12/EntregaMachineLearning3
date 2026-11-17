# Inteligencia musical: predicción de popularidad y segmentación de canciones (Spotify Tracks)

**Asignatura:** MLY1101 Machine Learning · **Evaluación Parcial N°3**
**Equipo:** Benjamin Neira, Gustavo Alvial y Matias Turra · **Fecha:** 16/11/2026

> Proyecto de Machine Learning que combina (1) **modelos supervisados de regresión** para predecir la popularidad de una canción (0-100) y (2) un modelo **no supervisado (K-Means)** para descubrir perfiles sonoros del catálogo. Los resultados de ambos enfoques se cruzan para extraer patrones accionables.

---

## Tabla de contenidos
1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del proyecto](#2-objetivos-del-proyecto)
3. [KPIs](#3-definición-de-kpis)
4. [Fuentes de datos](#4-fuentes-de-datos)
5. [Preparación y EDA](#5-preparación-y-análisis-exploratorio-eda)
6. [Metodología CRISP-DM](#6-metodología-crisp-dm)
7. [Desarrollo de los modelos](#7-desarrollo-de-los-modelos)
8. [Optimización de hiperparámetros y ensambles](#8-optimización-de-hiperparámetros-y-técnicas-de-ensamble)
9. [Evaluación de resultados](#9-evaluación-de-resultados)
10. [Segmentación y cruce con los modelos supervisados](#10-segmentación-y-cruce-con-los-modelos-supervisados)
11. [Conclusiones, limitaciones y recomendaciones](#11-conclusiones-limitaciones-y-recomendaciones)
12. [Reproducibilidad y estructura del proyecto](#12-reproducibilidad-y-estructura-del-proyecto)

---

## 1. Descripción del problema de negocio

Una plataforma de streaming musical administra un catálogo de decenas de miles de canciones de cientos de artistas y 114 géneros. Dos decisiones del negocio dependen de entender qué hace "exitosa" a una canción y cómo se agrupan sonoramente:

- **Priorización y promoción:** decidir qué canciones y artistas empujar en campañas, portadas y playlists editoriales. Hoy la popularidad solo se conoce *después* del lanzamiento.
- **Curación y recomendación:** armar playlists automáticas por similitud musical y mejorar las recomendaciones según el historial de escucha.

El problema se divide en dos preguntas:

1. **¿Se puede estimar la popularidad de una canción a partir de sus atributos?** (regresión supervisada)
2. **¿Existen perfiles sonoros naturales en el catálogo y cómo se relacionan con el éxito?** (segmentación no supervisada)

## 2. Objetivos del proyecto

**Objetivo general:** desarrollar y comparar soluciones de Machine Learning que estimen la popularidad de canciones y descubran perfiles musicales relevantes para la toma de decisiones.

**Objetivos específicos**
1. Preparar y analizar el dataset (calidad, duplicados, variables relevantes).
2. Implementar y comparar al menos dos modelos supervisados (aquí: regresión lineal como línea base y tres ensambles: Random Forest, LightGBM y XGBoost).
3. Optimizar hiperparámetros con validación cruzada y evaluar la capacidad de generalización.
4. Segmentar el catálogo con K-Means y caracterizar cada segmento.
5. Cruzar la segmentación con los modelos supervisados (importancia de variables, éxito y error por segmento).
6. Documentar sesgos, limitaciones y consideraciones éticas, y proponer recomendaciones.

## 3. Definición de KPIs

> Los umbrales son **metas de diseño propuestas por el equipo** (ajustables), contrastadas con lo obtenido.

### KPIs técnicos (modelo supervisado)
| KPI | Definición | Meta propuesta | Resultado (mejor modelo: Random Forest) |
|---|---|---|---|
| R² en test | Varianza de la popularidad explicada en datos no vistos | ≥ 0,55 | **0,579** |
| MAE en test | Error absoluto medio, en puntos de popularidad (0-100) | ≤ 9 | **8,43** |


### KPIs técnicos (segmentación)
| KPI | Definición | Meta propuesta | Resultado |
|---|---|---|---|
| Silhouette | Cohesión/separación de clusters | ≥ 0,5 | **≈ 0,59** (k = 3, leído del gráfico) |
| Interpretabilidad | Cada cluster con perfil sonoro distinguible | 3 perfiles | Instrumental, hablado, mayoritario |

### KPIs de negocio (propuestos para operar la solución)
| KPI | Uso |
|---|---|
| **Lift de éxito por segmento** (% de canciones en el top 20 % de popularidad vs. el promedio) | Decidir dónde invertir promoción |
| **Error del modelo por segmento** | Saber en qué segmentos confiar más en la predicción |
| **Cobertura de playlists automáticas** (% del catálogo asignado a un perfil) | Medir el alcance de la curación por similitud |

## 4. Fuentes de datos

- **Dataset:** *Spotify Tracks Dataset* (114.000 filas × 21 columnas, 114 géneros), cargado desde el repositorio del equipo.
- **Integridad:** el notebook valida el archivo con **SHA-256** (`b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`) antes de usarlo; si el hash no coincide, se detiene la ejecución.
- **Variables:**

| Grupo | Variables |
|---|---|
| Identificación / texto | `track_id`, `artists`, `album_name`, `track_name` |
| Objetivo | `popularity` (0-100) |
| Audio (0-1 salvo indicación) | `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`; `loudness` (dB), `tempo` (BPM) |
| Estructura musical | `key`, `mode`, `time_signature`, `duration_ms` |
| Contexto | `explicit`, `track_genre` |
| Creadas en el proyecto | `artista_principal`, `es_colaboracion`, `duration_s` |

## 5. Preparación y análisis exploratorio (EDA)

### 5.1 Calidad de datos
| Paso | Resultado |
|---|---|
| Valores nulos | 1 fila sin artista, álbum ni nombre → eliminada (113.999 filas) |
| Columna `Unnamed: 0` | Índice sin valor predictivo → eliminada |
| Duplicados `track_id + track_genre` | 894 filas implicadas → eliminadas |
| Misma canción en varios géneros | Se agrupa por `track_id` conservando la primera aparición → **89.740 canciones únicas** |
| Duración | `duration_ms` → `duration_s` |


### 5.2 Ingeniería de variables
- `artista_principal`: primer artista del campo `artists` (reduce la cardinalidad de combinaciones de intérpretes).
- `es_colaboracion`: 1 si la canción tiene más de un artista.
- `explicit`: booleano → entero.

### 5.3 Hallazgos
- **Popularidad:** media ≈ 33, desviación ≈ 22, con acumulación de canciones en 0 y distribución multimodal (no normal).
- **Correlaciones lineales (Spearman) con popularidad** son muy débiles: la mayor es `instrumentalness` (−0,078), seguida de `speechiness` (−0,068), `valence` (−0,042), `time_signature` (+0,036) y `loudness` (+0,033). **Ningún atributo de audio explica por sí solo la popularidad.**
- **Asociación categórica (V de Cramér vs. popularidad en quintiles):** `artista_principal` 0,79, `track_genre` 0,57, `es_colaboracion` 0,10. Los identificadores (`track_id`, `track_name`, `album_name`) muestran V altos (0,92-0,94) solo por ser casi únicos por fila; se **descartan** para evitar memorización.
- **Géneros:** pop-film y k-pop presentan medianas altas con baja dispersión; pop e hip-hop tienen alta dispersión (mayor riesgo).
- **Artistas:** los artistas con mayor popularidad promedio (p. ej. Olivia Rodrigo, 87,4) son casos de éxito extremo.

### 5.4 Preprocesamiento (dentro de `Pipeline` para evitar fuga de datos)
| Tipo | Variables | Tratamiento |
|---|---|---|
| Numéricas (12) | duración, danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature | Winsorización (2 %-98 %, ajustada solo en train) → imputación por media → `StandardScaler` |
| Binarias (3) | `es_colaboracion`, `explicit`, `mode` | Passthrough |
| Categóricas (2) | `artista_principal`, `track_genre` | **Target Encoding** de scikit-learn (`cv=5`, `smooth="auto"`), con validación cruzada interna para no filtrar el objetivo |
| Colinealidad | todas | Filtro de correlación > 0,9 (`CorrelationFilter`) |

Todo el preprocesamiento se ajusta **dentro de cada fold**, por lo que ni el Target Encoding ni el escalado ven datos de validación.

**Partición:** 80 % train (71.792) / 20 % test (17.948), `random_state = 42`.

## 6. Metodología (CRISP-DM)

| Fase | Aplicación en el proyecto |
|---|---|
| **1. Comprensión del negocio** | Dos decisiones: priorizar promoción y curar playlists (secciones 1-3) |
| **2. Comprensión de los datos** | Carga con verificación de hash, `info`/`describe`, nulos, duplicados, correlaciones, V de Cramér |
| **3. Preparación de los datos** | Limpieza, ingeniería de variables, pipelines, Target Encoding, partición train/test |
| **4. Modelado** | Regresión lineal, Random Forest, LightGBM, XGBoost; K-Means + PCA |
| **5. Evaluación** | Validación cruzada de 5 folds, R² y MAE en train/CV/test, análisis por segmento |
| **6. Despliegue** | Entregables reproducibles; recomendaciones de uso (curación, priorización). Un despliegue productivo requiere monitoreo (ver sección 11) |

## 7. Desarrollo de los modelos de aprendizaje automático

### 7.1 Modelos supervisados (regresión)
| Modelo | Rol | Familia |
|---|---|---|
| Regresión lineal | Línea base | Lineal |
| Random Forest | Captura no linealidades e interacciones (género × artista) | Ensamble **bagging** |
| LightGBM | Gradient boosting eficiente | Ensamble **boosting** |
| XGBoost | Gradient boosting regularizado | Ensamble **boosting** |

**Todos usan exactamente el mismo preprocesamiento** (`pipeline_regresion`, incluyendo `artista_principal`), por lo que la comparación es justa. Se mantiene el artista por decisión del equipo; el riesgo asociado se discute en la sección 11.

Los modelos base sin ajustar dieron: regresión lineal R² test 0,536 (train 0,671) y Random Forest R² test 0,568 con train 0,782 (sobreajuste).

### 7.2 Modelo no supervisado
- **Variables:** 9 atributos de audio (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `tempo`, `valence`).
- **Se excluyen** `popularity` (para no segmentar por el objetivo), género y artista (se usan después para *caracterizar*), y duración, tonalidad, modo, compás y `explicit`.
- **Preprocesamiento:** imputación por mediana y `RobustScaler` (por la presencia de atípicos).
- **Algoritmo:** K-Means (`random_state = 67`, `n_init = 10`); k elegido con codo (`KneeLocator`) y silhouette; PCA de 2 componentes solo para visualizar.

## 8. Optimización de hiperparámetros y técnicas de ensamble

### 8.1 Estrategia
- **Validación:** `cv = 5` para todos los modelos; selección (`refit`) por **R²**; se registra también MAE.
- **Random Forest, regresión lineal y LightGBM:** `GridSearchCV`. **XGBoost:** `RandomizedSearchCV` con 40 iteraciones (el espacio es demasiado grande para una búsqueda exhaustiva).

| Modelo | Espacio de búsqueda | Mejores parámetros |
|---|---|---|
| Random Forest | `max_depth` [6, 8, 12, 16], `min_samples_leaf` [3, 10], `max_features` [0,5; 1,0], `n_estimators` 200 (16 combinaciones) | `max_depth = 16`, `min_samples_leaf = 3`, `max_features = 0,5` |
| LightGBM | `n_estimators` [300, 600], `learning_rate` [0,03; 0,1], `num_leaves` [31, 63], `min_child_samples` [20, 50], `smooth` [auto, 20] (32 combinaciones) | `n_estimators = 600`, `learning_rate = 0,03`, `num_leaves = 63`, `min_child_samples = 20`, `smooth = auto` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `subsample`, `colsample_bytree`, `gamma` (40 de ellas muestreadas) | `n_estimators = 800`, `learning_rate = 0,03`, `max_depth = 6`, `min_child_weight = 10`, `gamma = 2`, `subsample = 0,9`, `colsample_bytree = 0,6`, `reg_lambda = 1`, `reg_alpha = 0` |
| Regresión lineal | Parámetros del preprocesamiento (winsorización, `smooth`, umbral de colinealidad) | umbral 0,8, `smooth = auto`, sin winsorización |

### 8.2 Efecto de la optimización
| Random Forest | R² train | R² test | MAE test |
|---|---|---|---|
| Base (sin ajustar) | 0,782 | 0,568 | 8,515 |
| Optimizado | 0,759 | **0,579** | **8,431** |

La optimización **mejoró el R² test en ~0,01, redujo el MAE y recortó el sobreajuste** (brecha train-test de 0,214 a 0,180), con regularización por profundidad y tamaño mínimo de hoja.

### 8.3 Aporte de los ensambles
Los tres ensambles superan a la regresión lineal en unos **0,034-0,043 de R²** y **0,55-0,75 puntos de MAE** en test, lo que evidencia relaciones **no lineales e interacciones** (por ejemplo, el efecto del artista depende del género) que el modelo lineal no captura.
- **Bagging** (Random Forest) reduce la varianza promediando muchos árboles profundos.
- **Boosting** (LightGBM, XGBoost) reduce el sesgo construyendo árboles secuenciales que corrigen errores previos, con regularización y submuestreo.

## 9. Evaluación de resultados

### 9.1 Métricas
**R²** (proporción de varianza explicada) y **MAE** (error medio en puntos de popularidad, fácil de comunicar al negocio).

### 9.2 Comparación de modelos (5 folds de CV + test)
| Modelo | R² train | R² CV | ± std CV | R² test | MAE CV | MAE test | Brecha R² (train-test) |
|---|---|---|---|---|---|---|---|
| **Random Forest (bagging)** | 0,762 | **0,575** | 0,008 | **0,579** | **8,502** | **8,431** | 0,184 |
| LightGBM (boosting) | 0,754 | 0,570 | 0,009 | 0,574 | 8,612 | 8,546 | 0,181 |
| XGBoost (boosting) | 0,745 | 0,567 | 0,009 | 0,570 | 8,708 | 8,636 | 0,175 |
| Regresión lineal | 0,673 | 0,528 | 0,009 | 0,536 | 9,315 | 9,183 | 0,137 |



### 9.3 Capacidad de generalización
- El R² de **CV y de test coinciden** en los cuatro modelos (diferencias de 0,004-0,008) y la desviación entre folds es baja (≤ 0,009): los modelos generalizan de forma estable y no se ajustaron al test.
- La **brecha train-test** (0,14-0,18) no debe leerse solo como sobreajuste: una parte proviene de que el Target Encoding favorece al conjunto de entrenamiento. Por eso se prioriza CV y test para decidir.
- Las operaciones de preprocesamiento están dentro de los folds, evitando fuga de datos.

### 9.4 Selección y justificación de la solución óptima
**Modelo seleccionado: Random Forest optimizado.**
1. Obtiene el **mejor desempeño en las cuatro métricas de generalización** (R² y MAE, en CV y en test).
2. CV y test son consistentes.
3. Su ventaja sobre LightGBM y XGBoost es **pequeña** (≤ 0,008 de R² en CV, del orden de la variabilidad entre folds), por lo que los tres ensambles pueden considerarse de desempeño comparable; se escoge Random Forest por ser consistentemente el mejor y por la simplicidad de ajustar su configuración.
4. Supera claramente a la línea base lineal.

### 9.5 Importancia de variables (permutación, caída de R² en test)
| Variable | Random Forest | LightGBM | XGBoost |
|---|---|---|---|
| `artista_principal` | 0,661 | 0,652 | 0,636 |
| `track_genre` | 0,145 | 0,151 | 0,151 |
| `loudness` | 0,008 | 0,007 | 0,007 |
| `energy` | 0,008 | 0,005 | 0,005 |
| `instrumentalness` | 0,008 | 0,005 | 0,005 |
| `duration_s` | 0,007 | 0,006 | 0,006 |
| `acousticness` | 0,006 | 0,004 | 0,004 |
| resto | < 0,006 | < 0,005 | < 0,005 |

**El modelo se apoya casi por completo en el artista y el género**; cada atributo acústico aporta menos de un punto de R². Los tres modelos coinciden en el orden de importancia.



## 10. Segmentación y cruce con los modelos supervisados

### 10.1 Elección de k
El silhouette es ≈ 0,59 para k = 2 y k = 3, y cae a ≈ 0,47 en k = 4 y a ≈ 0,34 en k = 5. Se elige **k = 3** (coincide con el codo de `KneeLocator`): mantiene la separación casi igual que k = 2 y añade un segmento con significado propio. Las 2 componentes del PCA retienen el 80,7 % de la varianza, por lo que la proyección 2D representa bien los datos.

![Clusters en PCA](images/clusters_pca.png)

### 10.2 Perfil de cada cluster
| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| **Nombre propuesto** | Núcleo / mayoritario | Instrumental y relajación | Voz hablada / comedia |
| Canciones | 67.030 (74,7 %) | 17.460 (19,5 %) | 5.250 (5,9 %) |
| `instrumentalness` | 0,02 | **0,81** | 0,02 |
| `speechiness` | 0,07 | 0,06 | **0,44** |
| `loudness` (dB) | −7,4 | **−12,8** | −7,8 |
| `energy` | 0,65 | 0,55 | 0,69 |
| `danceability` | 0,57 | 0,50 | 0,64 |
| `valence` | 0,51 | **0,30** | 0,52 |
| `liveness` | 0,22 | 0,18 | **0,32** |
| % explícitas | 7,8 | 2,5 | **39,0** |
| Género más frecuente | *forro* (poco representativo) | *study* | *comedy* |
| Géneros distintos / HHI de géneros | 113 / 0,010 | 104 / 0,028 | 111 / 0,044 |

**Lectura:** los clusters se definen sobre todo por dos variables: `instrumentalness` (eta² = 0,93) y `speechiness` (eta² = 0,61); `loudness` aporta algo (0,16) y el resto casi nada. El **cluster 0** es la música "vocal convencional" con gran diversidad de géneros (HHI ≈ 0,010, casi uniforme: el género más frecuente no lo caracteriza). El **cluster 1** es música instrumental, suave y menos alegre (estudio, relajación). El **cluster 2** es contenido hablado (género más frecuente: comedia), con alto porcentaje de contenido explícito y la mayor concentración de géneros y artistas.

> Nota: el cluster 2 **no** es un segmento "de alta energía y baile": su `energy` y `danceability` son solo ligeramente superiores a las del cluster 0 y su rasgo distintivo es la palabra hablada.

### 10.3 Éxito y popularidad por segmento
| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Popularidad media / mediana | 34,6 / 36 | 28,1 / 24 | 32,2 / 27 |
| % en el top 20 % de popularidad | 23,1 | 13,4 | 17,6 |
| **Lift de éxito** (vs. promedio) | **1,10** | 0,64 | 0,84 |
| % con popularidad 0 | **11,6** | 7,6 | 7,0 |

El cluster 0 concentra más éxitos, pero también más canciones con popularidad 0: es el segmento más **polarizado**. El cluster instrumental tiene un lift de 0,64 (menos probabilidad de éxito masivo).

### 10.4 ¿Las variables que separan clusters son las que predicen popularidad?
Los clusters se definen por `instrumentalness` y `speechiness`, pero la importancia de esas variables en los modelos es baja. La correlación de Spearman entre el ranking de separación y el de importancia es de **0,35-0,40** (solo 9 variables, por lo que es un indicador descriptivo, no una prueba). Conclusión: **segmentar por sonido no es lo mismo que predecir éxito**; los clusters sirven para *curar y recomendar por similitud*, no para estimar popularidad. Lo que predice popularidad es el artista y el género.

### 10.5 ¿Cambian los factores de éxito dentro de cada cluster?
Diferencia estandarizada entre canciones del top 20 % y el resto, dentro de cada cluster (asociación, no causalidad; diferencias en general < 0,5 desviaciones estándar):

| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Más frecuente en éxitos | más `danceability` (+0,16), más `loudness` (+0,12) | más `acousticness` (+0,37) | más `loudness` (+0,22), más `danceability` (+0,18) |
| Menos frecuente en éxitos | menos `liveness` (−0,21), menos `acousticness` (−0,15), menos `valence` (−0,14) | menos `loudness` (−0,46), menos `energy` (−0,44), menos `liveness` (−0,23) | menos `speechiness` (−0,52), menos `liveness` (−0,42), menos `acousticness` (−0,37) |

Dentro del cluster instrumental, el éxito se asocia con música **más acústica, suave y baja**, lo opuesto a lo que ocurre en los otros dos clusters (donde más volumen se asocia con éxito). Es decir, **el "sonido del éxito" depende del segmento**, y un modelo global promedia efectos contrarios.

### 10.6 Peso del artista y género por cluster
| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Artistas únicos | 14.166 | 5.005 | 2.268 |
| Canciones por artista | 4,7 | 3,5 | 2,3 |
| % de canciones del top 10 de artistas | 2,9 | 5,8 | 6,9 |
| η² popularidad ~ género | 0,295 | **0,468** | 0,392 |
| η² popularidad ~ artista (sin ajustar) | 0,669 | 0,784 | 0,818 |

El género explica más la popularidad en el cluster instrumental (η² = 0,47). Los η² por artista **están inflados por la alta cardinalidad** (pocas canciones por artista), sobre todo en el cluster 2; deben ajustarse antes de compararse.

### 10.7 ¿Dónde falla más el modelo? (MAE en test por cluster)
| Modelo | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Regresión lineal | 9,64 | 7,95 | 7,33 |
| Random Forest | 8,85 | 7,28 | 6,80 |
| LightGBM | 8,97 | 7,41 | 6,83 |
| XGBoost | 9,07 | 7,47 | 6,94 |
| Canciones de test | 13.466 | 3.426 | 1.056 |

- El error es mayor en el cluster 0 y menor en los otros; sin embargo, la menor popularidad y dispersión de los clusters 1 y 2 puede explicar parte de ello (se recomienda comparar el MAE con la desviación estándar de la popularidad de cada cluster).
- Los ensambles mejoran sobre el modelo lineal en los tres clusters (la mayor ganancia, de ≈ 0,8 puntos, ocurre en el cluster 0).
- Sesgo medio (predicho − real): ≈ 0 en los clusters 0 y 1; **+0,7 a +0,9 en el cluster 2** (el modelo sobrestima levemente la popularidad del contenido hablado).

### 10.8 Recomendaciones derivadas de la segmentación
- **Cluster 0 (núcleo):** lanzamientos masivos y campañas generales; es donde está la mayor probabilidad de éxito, pero también de popularidad nula, por lo que conviene combinar con señales de artista y género.
- **Cluster 1 (instrumental/relajación):** playlists por momento del día y actividad (estudio, concentración, descanso). Dentro del segmento, priorizar versiones **acústicas y suaves**.
- **Cluster 2 (hablado):** tratarlo como un formato aparte (podcast, comedia, spoken word) con filtros de contenido explícito (39 %) y sin mezclarlo con playlists musicales.

## 11. Conclusiones, limitaciones y recomendaciones

### 11.1 Conclusiones
1. Los ensambles superan a la regresión lineal (R² test 0,570-0,579 vs. 0,536; MAE 8,4-8,6 vs. 9,2). **Random Forest optimizado** es el mejor modelo: R² test 0,579 y MAE 8,43 sobre una escala de 0 a 100.
2. Los modelos generalizan de forma estable (CV ≈ test; desviación entre folds ≤ 0,009).
3. La popularidad depende sobre todo de **quién es el artista y de qué género es**; los atributos acústicos aportan muy poco por sí solos.
4. El catálogo se divide en tres perfiles sonoros claros (silhouette ≈ 0,59), definidos principalmente por instrumentalidad y cantidad de voz hablada.
5. Los patrones de éxito cambian por segmento, lo que justifica estrategias de recomendación diferenciadas.
6. El R² cercano a 0,58 deja ~42 % de la variabilidad sin explicar: faltan factores como marketing, playlists editoriales, fecha de lanzamiento o viralidad.

### 11.2 Limitaciones
- **Fuga potencial por artista:** `artista_principal` se mantuvo por decisión del equipo, pero ~**80,1 %** de los artistas del test también están en train. El modelo aprende en buena medida "quién es el artista", por lo que el desempeño es **optimista para artistas nuevos**. Una validación por grupos de artistas daría una estimación más conservadora.
- **Un solo género por canción:** al agrupar por `track_id` y conservar la primera fila, se pierde información de canciones presentes en varios géneros (y su popularidad puede diferir entre ellas).
- **Popularidad = 0:** hay acumulación de ceros que pueden reflejar canciones sin datos y no demanda real.
- **Hiperparámetros en el borde del espacio:** varios óptimos quedaron en el límite de la grilla (Random Forest: `max_depth = 16`, `min_samples_leaf = 3`; XGBoost: `n_estimators = 800`, `max_depth = 6`), por lo que un espacio más amplio podría mejorar levemente el desempeño.
- **Diferencias pequeñas entre ensambles:** no se realizó una prueba estadística formal entre Random Forest, LightGBM y XGBoost; se interpreta como desempeño comparable.
- **Los patrones por cluster son asociativos:** no implican causalidad y no controlan el efecto del artista. Los η² por artista requieren ajuste.
- **Sin dimensión temporal:** el dataset es una foto; la popularidad cambia en el tiempo.

### 11.3 Consideraciones éticas y sesgos
- **Sesgo de visibilidad:** el modelo tiende a asignar popularidad alta a artistas ya populares, lo que puede **reforzar un círculo vicioso** y perjudicar a artistas emergentes si se usa para recomendar o priorizar contenido.
- **Sesgo geográfico y de género musical:** la muestra no es necesariamente representativa de todos los mercados ni de todos los estilos.
- **Contenido explícito:** el segmento de voz hablada concentra el 39 % de contenido explícito; debe tratarse con controles de edad y preferencias del usuario.
- **Uso responsable:** no debe ser el único criterio para decisiones de inversión o exposición de artistas. Para **descubrir talento nuevo** se recomienda un modelo alternativo **sin la variable artista**.
- **Datos personales:** no se utilizan datos de usuarios, solo atributos de canciones.

### 11.4 Recomendaciones para el negocio
1. Usar el modelo para **priorizar y dar un puntaje inicial** de popularidad esperada, no como decisión única.
2. Para **artistas nuevos**, entrenar y comparar una versión sin `artista_principal` y validarla con `GroupKFold` por artista.
3. Utilizar la **segmentación para curar playlists** y recomendaciones por similitud, con estrategias distintas por perfil.
4. Incorporar variables faltantes (fecha de lanzamiento, playlists editoriales, métricas sociales) para superar el techo de R² ≈ 0,58.
5. **Monitorear** el error por segmento y reentrenar periódicamente.

---

## 12. Reproducibilidad y estructura del proyecto

```
proyecto/
├── README.md                          # Este informe
├── requirements.txt
├── data/
│   ├── Spotify_Tracks_Dataset.csv     # (o descarga con verificación de hash)
│   ├── importancias_modelos.csv       # salida de supervisado → entrada de segmentación
│   ├── predicciones_test.csv
│   └── tabla_comparacion_modelos.csv
├── notebooks/
│   ├── 01_supervisado_ensambles.ipynb
│   └── 02_segmentacion.ipynb
├── models/
│   └── random_forest_final.joblib
└── images/
```

**Orden de ejecución**
1. Ejecutar `01_supervisado_ensambles.ipynb` completo; genera los CSV de `data/` (importancias y predicciones de test).
2. Subir esos CSV a `data/` del repositorio (o dejarlos en la carpeta local).
3. Ejecutar `02_segmentacion.ipynb`, que los lee para cruzar ambos análisis.

**Semillas:** `random_state = 42` (supervisado) y `67` (K-Means).
**Dependencias principales:** `pandas`, `numpy`, `scikit-learn ≥ 1.3` (por `TargetEncoder`), `xgboost`, `lightgbm`, `category_encoders`, `mlxtend`, `kneed`, `matplotlib`, `seaborn`, `scipy`.
