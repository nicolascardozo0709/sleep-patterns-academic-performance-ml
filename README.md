# Predicción de Alto Rendimiento Académico (GPA) a partir de Hábitos de Sueño (Fitbit)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **Asignatura:** Introducción a la Inteligencia Artificial  
> **Institución:** Pontificia Universidad Javeriana — Departamento de Ingeniería de Sistemas  
> **Docente:** Enrique González, PhD  
> **Proyecto:** Proyecto 3 — Modelos de Clasificación Supervisada y Diseño Experimental  

---

## 1. Resumen Ejecutivo del Proyecto

Este proyecto investiga la relación causal y predictiva entre los **patrones biométricos objetivos de sueño** (registrados longitudinalmente mediante wearables Fitbit) y el **rendimiento académico universitario** medido a través del promedio acumulado del semestre (`term_gpa`).

El problema se formula como una **clasificación binaria supervisada**:
$$\text{Clase 1: Alto Rendimiento} \quad (\text{term\_gpa} \ge 3.5)$$
$$\text{Clase 0: Rendimiento Estándar/Bajo} \quad (\text{term\_gpa} < 3.5)$$

### Contexto Científico y Benchmark
El trabajo se fundamenta en el dataset abierto de la Universidad Carnegie Mellon (CMU) publicado en:
* **Artículo PNAS (2023):** Creswell, J. D., Tumminia, M. J., Price, S., et al. *"Nightly sleep duration predicts grade point average in the first year of college"*. *Proceedings of the National Academy of Sciences (PNAS)*, 120(8), e2209123120.

A diferencia del estudio original (que utilizó modelos econométricos y lineales tradicionales), este proyecto:
1. **Reformula el problema como clasificación temprana:** Permite identificar a estudiantes en riesgo académico versus aquellos con alta probabilidad de honores.
2. **Evalúa fronteras no lineales:** Modela interacciones multidimensionales mediante **Random Forest (RF)**, **Support Vector Machines (SVM)** y **Perceptrón Multicapa (MLP)**.
3. **Garantiza cero fuga de datos (*zero data leakage*):** Emplea un protocolo riguroso con `Pipeline` y validación cruzada estratificada en 5 folds.

---

## 2. Estructura del Repositorio

```text
├── data/
│   ├── cmu-sleep.csv                 <- Dataset original crudo (634 filas, 15 variables)
│   └── cmu_sleep_limpio.csv          <- Dataset preprocesado con target binario
├── resultados/
│   ├── tabla_pruebas_preliminares.csv            <- Barrido preliminar de modelos
│   ├── tabla_experimento_Random_Forest.csv       <- 9 casos experimentales RF
│   ├── tabla_experimento_SVM.csv                 <- 9 casos experimentales SVM
│   ├── tabla_experimento_MLP.csv                 <- 9 casos experimentales MLP
│   ├── tabla_comparativa_mejores_modelos.csv     <- Cuadro comparativo final
│   └── figuras/
│       ├── fig1_eda_distribuciones_y_correlacion.png
│       ├── fig2_diseno_experimental_esquema.png
│       ├── fig_resultados_Random_Forest.png
│       ├── fig_resultados_SVM.png
│       ├── fig_resultados_MLP.png
│       └── fig6_comparacion_y_matrices_confusion.png
├── proyecto3_sueno_gpa.py            <- Script de ejecución completa en Python (596 líneas)
├── Proyecto_3_Sueno_vs_GPA.ipynb     <- Jupyter Notebook interactivo celda a celda
├── requirements.txt                  <- Dependencias del entorno
└── README.md
```

---

## 3. Dataset y Preprocesamiento

* **Muestra:** $N = 634$ estudiantes de primer año universitario distribuidos en 5 cohortes académicas.
* **Instrumentación:** Relojes biométricos Fitbit Flex y Fitbit Charge HR utilizados durante todo el semestre académico.
* **Balance de Clases:** Moderado y sano: Clase 1 ($57.89\%$, 367 estudiantes) y Clase 0 ($42.11\%$, 267 estudiantes).

### Reglas de Limpieza y Transformación
1. **Espacios en blanco:** Reemplazo de espacios vacíos camuflados (`' '`) por `np.nan`.
2. **Corrección de dominio:** Depuración de la variable `demo_firstgen` restringiéndola estrictamente al dominio binario $\{0, 1\}$.
3. **Corrección de asimetría (*skewness*):** Transformación $\text{log\_bedtime\_mssd} = \ln(1 + \text{bedtime\_mssd})$, reduciendo el *skewness* de **8.24** a **1.15**, estabilizando la varianza.

---

## 4. Metodología y Diseño Experimental

### Protocolo Anti-Data Leakage
Para evitar que la media y la desviación estándar del conjunto de prueba contaminen el entrenamiento, todo el preprocesamiento se encapsuló en un `Pipeline` de Scikit-Learn ejecutado estrictamente dentro de cada fold:
1. `SimpleImputer(strategy='median')`
2. `StandardScaler()`
3. Clasificador (`RandomForestClassifier`, `SVC`, `MLPClassifier`)

### Esquema de Validación y Métricas
* **Estrategia:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
* **Métrica Principal:** **F1-Score Macro** (media armónica no ponderada entre clases, evalúa equidad entre falsos positivos y falsos negativos).
* **Métricas Secundarias:** Accuracy, Precision Macro, Recall Macro, ROC-AUC y Desviación Estándar ($\sigma$).

### Matriz Experimental 3×3 (27 Experimentos Totales)
Se variaron 3 hiperparámetros clave por modelo en 3 niveles (Bajo, Defecto, Alto):
* **Random Forest:** `n_estimators` (5, 100, 500), `max_depth` (1, 10, 30), `min_samples_leaf` (1, 5, 80).
* **SVM:** $C$ (0.01, 1.0, 100.0), $\gamma$ (0.001, `'scale'`, 2.0), $\text{Kernel}$ (`'linear'`, `'rbf'`, `'poly'` $d=3$).
* **MLP:** `hidden_layer_sizes` ((3,), (100,), (300, 150)), `learning_rate_init` (0.0001, 0.001, 0.5), `max_iter` (20, 200, 1000).

---

## 5. Resultados Principales

### Comparativa de los Mejores Modelos por Técnica

| Técnica | Caso | Hiperparámetro Clave | F1-Macro CV | Desv. ($\sigma$) | Accuracy | Precision | Recall | ROC-AUC | Matriz de Confusión Acumulada |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **C3** | **`n_estimators=500`** | **0.7446** | **0.0371** | **75.24%** | **0.7462** | **0.7438** | **0.8174** | **TN=184, FP=83, FN=74, TP=293** |
| **MLP** | C2 | `hidden_layer_sizes=(100,)` | 0.7151 | 0.0373 | 72.40% | 0.7168 | 0.7146 | 0.8007 | TN=175, FP=92, FN=83, TP=284 |
| **SVM** | C2 | $C=1.0$ (Kernel RBF) | 0.7127 | 0.0451 | 72.56% | 0.7199 | 0.7104 | 0.7946 | TN=164, FP=103, FN=71, TP=296 |

### Conclusiones Técnicas
1. **Modelo Ganador:** **Random Forest (C3: 500 estimadores)** demostró el mejor desempeño general, con un **Recall de 79.84%** en la detección de estudiantes de alto GPA y la menor desviación estándar ($\sigma = 0.0371$).
2. **Relevancia del Sueño:** La irregularidad circadiana (`log_bedtime_mssd`) y la duración del sueño (`TotalSleepTime`) aportan información predictiva crítica y no lineal para el desempeño académico.
3. **Resistencia al Sobreajuste:** Random Forest ratificó el Teorema de Breiman: aumentar $B \to \infty$ no sobreajusta el modelo, sino que estabiliza la varianza asintóticamente en $\rho\sigma^2$.

---

## 6. Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/nicolascardozo0709/sleep-patterns-academic-performance-ml.git
cd sleep-patterns-academic-performance-ml
```

### 2. Entorno virtual e instalación de dependencias
```bash
python -m venv .venv
source .venv/bin/activate  # En Linux / macOS
# .venv\Scripts\activate   # En Windows

pip install -r requirements.txt
```

### 3. Reproducir el proyecto completo
Para ejecutar todo el flujo (limpieza, 27 experimentos, tablas y figuras):
```bash
python proyecto3_sueno_gpa.py
```
O explorar el cuaderno Jupyter paso a paso:
```bash
jupyter notebook Proyecto_3_Sueno_vs_GPA.ipynb
```

---

## Autores y Créditos
* Nicolas Cardozo Mejia
* Integrantes del Grupo 4 — Introducción a la Inteligencia Artificial
* Pontificia Universidad Javeriana (Bogotá, Colombia)
