"""
================================================================================
PROYECTO 3 - INTRODUCCIÓN A LA INTELIGENCIA ARTIFICIAL
Pontificia Universidad Javeriana - Departamento de Ingeniería de Sistemas
--------------------------------------------------------------------------------
Problema de Clasificación:
    Predicción de Alto Rendimiento Académico Universitario (term_gpa >= 3.5)
    a partir de Datos Biométricos de Sueño (Fitbit) e Historial Académico.

Dataset (Benchmark):
    "CMU Sleep / Sleep vs. GPA in College" (Creswell et al., PNAS 2023)
    Kaggle: https://www.kaggle.com/datasets/mexwell/cmu-sleep
    CMU S&DS: https://cmustatistics.github.io/data-repository/psychology/cmu-sleep.html

Técnicas Evaluadas (Fase 2 y Fase 3):
    1. Random Forest (RF)
    2. Support Vector Machines (SVM)
    3. Redes Neuronales - Perceptrón Multicapa (MLP)

Diseño Experimental (Fase 3):
    - Variable Dependiente: F1-Score Macro (derivada de la matriz de confusión),
      acompañada de Accuracy, Precision, Recall y ROC-AUC.
    - Variables Independientes: 3 hiperparámetros por técnica x 3 niveles
      (Bajo, Defecto/Base, Alto) = 9 casos por técnica (27 casos en total).
    - Validación: Validación Cruzada Estratificada (Stratified 5-Fold CV)
      con preprocesamiento dentro de Pipeline (cero fuga de datos / data leakage).
================================================================================
"""

import os
import warnings
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Configuración estética global para gráficos listos para PowerPoint
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "resultados")
FIG_DIR = os.path.join(OUT_DIR, "figuras")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "cmu-sleep.csv")
CSV_URL = "https://cmustatistics.github.io/data-repository/data/cmu-sleep.csv"

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
UMBRAL_GPA = 3.5  # Lista de Honor / Alto Rendimiento Universitario (>= 3.5 sobre 4.0)


# ==============================================================================
# FASE 2A: CARGA, ANÁLISIS EXPLORATORIO (EDA) Y LIMPIEZA DE DATOS
# ==============================================================================
def cargar_y_limpiar_datos():
    print("=" * 80)
    print("FASE 2: CARGA, INSPECCIÓN Y LIMPIEZA DE DATOS (cmu-sleep.csv)")
    print("=" * 80)

    if not os.path.exists(CSV_PATH):
        print(f"Descargando dataset oficial desde {CSV_URL}...")
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)

    # 1. Lectura cruda para detectar nulos ocultos como espacios en blanco (' ')
    df_raw = pd.read_csv(CSV_PATH)
    print(f"[1] Dimensiones originales: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")
    print(f"[2] Nulos aparentes en lectura cruda: {df_raw.isnull().sum().sum()}")

    # 2. Lectura reconociendo espacios en blanco (' ') como NaN reales
    df = pd.read_csv(CSV_PATH, na_values=[" ", ""])
    nulos_reales = df.isnull().sum()
    print("\n[3] Valores faltantes REALES detectados (ocultos como espacios ' '):")
    print(nulos_reales[nulos_reales > 0].to_string())

    # 3. Detección y corrección de valores atípicos/inválidos de dominio
    n_invalid_firstgen = (df["demo_firstgen"] == 2).sum()
    print(f"\n[4] Anomalía de dominio detectada: {n_invalid_firstgen} registro(s) con demo_firstgen == 2 (dominio binario 0/1).")
    df["demo_firstgen"] = df["demo_firstgen"].replace(2, np.nan)

    # 4. Ingeniería de características (Feature Engineering)
    # bedtime_mssd presenta fuerte asimetría positiva (skew > 8), se aplica log1p
    skew_antes = df["bedtime_mssd"].skew()
    df["log_bedtime_mssd"] = np.log1p(df["bedtime_mssd"])
    skew_despues = df["log_bedtime_mssd"].skew()
    print(f"[5] Transformación log1p(bedtime_mssd): asimetría reducida de {skew_antes:.2f} a {skew_despues:.2f}")

    # 5. Creación de variable objetivo de CLASIFICACIÓN BINARIA
    # 1 = Alto Rendimiento (term_gpa >= 3.5), 0 = Rendimiento Estándar/Bajo (term_gpa < 3.5)
    df["Alto_Rendimiento"] = (df["term_gpa"] >= UMBRAL_GPA).astype(int)
    conteo = df["Alto_Rendimiento"].value_counts().sort_index()
    print(f"\n[6] Distribución de la Variable Objetivo (Alto_Rendimiento: term_gpa >= {UMBRAL_GPA}):")
    print(f"    Clase 0 (Estándar / Riesgo, < {UMBRAL_GPA}): {conteo[0]} ({conteo[0]/len(df)*100:.1f}%)")
    print(f"    Clase 1 (Alto Rendimiento, >= {UMBRAL_GPA}): {conteo[1]} ({conteo[1]/len(df)*100:.1f}%)")

    # 6. Selección de variables predictoras (se eliminan IDs y term_units por escalas heterogéneas entre universidades;
    #    se conserva Zterm_units_ZofZ que ya normaliza la carga académica por cohorte)
    num_features = [
        "TotalSleepTime",
        "log_bedtime_mssd",
        "midpoint_sleep",
        "daytime_sleep",
        "frac_nights_with_data",
        "cum_gpa",
        "Zterm_units_ZofZ",
    ]
    cat_bin_features = [
        "demo_race",
        "demo_gender",
        "demo_firstgen",
    ]
    features = num_features + cat_bin_features

    X = df[features].copy()
    y = df["Alto_Rendimiento"].copy()

    # Guardar versión limpia descriptiva para verificación / RapidMiner si se requiere
    df_export = X.copy()
    for c in num_features:
        df_export[c] = df_export[c].fillna(df_export[c].median())
    for c in cat_bin_features:
        df_export[c] = df_export[c].fillna(df_export[c].mode()[0])
    df_export["Alto_Rendimiento"] = y
    df_export.to_csv(os.path.join(DATA_DIR, "cmu_sleep_limpio.csv"), index=False)
    print(f"[7] Dataset limpio exportado a: {os.path.join(DATA_DIR, 'cmu_sleep_limpio.csv')}")

    generar_graficos_eda(df, features)
    return df, X, y, num_features, cat_bin_features


def generar_graficos_eda(df, features):
    """Genera figura resumen de EDA (distribución GPA, sueño por clase y matriz de correlación)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Subplot 1: Distribución de term_gpa y umbral de clasificación
    sns.histplot(df["term_gpa"], bins=25, kde=True, color="#2563EB", ax=axes[0])
    axes[0].axvline(UMBRAL_GPA, color="#DC2626", linestyle="--", linewidth=2, label=f"Umbral = {UMBRAL_GPA}")
    axes[0].set_title("Distribución de GPA del Semestre (term_gpa)", fontweight="bold")
    axes[0].set_xlabel("GPA Semestral (0.0 - 4.0)")
    axes[0].set_ylabel("Cantidad de Estudiantes")
    axes[0].legend()

    # Subplot 2: Horas de sueño nocturno vs Clase de Rendimiento
    df_plot = df.copy()
    df_plot["Horas_Sueno"] = df_plot["TotalSleepTime"] / 60.0
    df_plot["Clase"] = df_plot["Alto_Rendimiento"].map({0: "Estándar/Riesgo (<3.5)", 1: "Alto Rendimiento (>=3.5)"})
    sns.boxplot(data=df_plot, x="Clase", y="Horas_Sueno", palette=["#F59E0B", "#10B981"], ax=axes[1])
    axes[1].set_title("Horas de Sueño Nocturno (Fitbit) por Clase", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Horas promedio de sueño / noche")

    # Subplot 3: Correlación con term_gpa y variables principales
    corr_cols = ["term_gpa", "cum_gpa", "TotalSleepTime", "midpoint_sleep", "daytime_sleep", "log_bedtime_mssd", "Zterm_units_ZofZ"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": 0.8}, ax=axes[2])
    axes[2].set_title("Matriz de Correlación (Variables Clave)", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=35, labelsize=8)
    axes[2].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    ruta = os.path.join(FIG_DIR, "fig1_eda_distribuciones_y_correlacion.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"    -> Figura EDA guardada: {ruta}")


# ==============================================================================
# PIPELINE DE PREPROCESAMIENTO (CERO DATA LEAKAGE DENTRO DE K-FOLD CV)
# ==============================================================================
def construir_pipeline(modelo, num_features, cat_bin_features, escalar=True):
    """
    Construye un Pipeline de scikit-learn donde la imputación y la normalización
    se ajustan únicamente con el fold de entrenamiento en cada iteración de CV.
    """
    if escalar:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_bin_features),
        ]
    )

    return Pipeline([
        ("prep", preprocessor),
        ("clf", modelo),
    ])


def evaluar_modelo(pipe, X, y):
    """Evalúa un pipeline mediante Stratified 5-Fold CV y retorna métricas clave."""
    scoring = {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "roc_auc": "roc_auc",
    }
    res = cross_validate(pipe, X, y, cv=CV, scoring=scoring, n_jobs=-1)
    return {
        "F1_Macro": res["test_f1_macro"].mean(),
        "F1_Std": res["test_f1_macro"].std(),
        "Accuracy": res["test_accuracy"].mean(),
        "Precision_Macro": res["test_precision_macro"].mean(),
        "Recall_Macro": res["test_recall_macro"].mean(),
        "ROC_AUC": res["test_roc_auc"].mean(),
    }


# ==============================================================================
# FASE 3A: PRUEBAS PRELIMINARES DE SENSIBILIDAD DE HIPERPARÁMETROS
# ==============================================================================
def ejecutar_pruebas_preliminares(X, y, num_features, cat_bin_features):
    print("\n" + "=" * 80)
    print("FASE 3A: PRUEBAS PRELIMINARES DE SENSIBILIDAD DE HIPERPARÁMETROS")
    print("=" * 80)
    registros = []

    pruebas = [
        # Random Forest
        ("Random Forest", False, lambda **kw: RandomForestClassifier(random_state=SEED, **kw), "n_estimators", [5, 15, 50, 100, 300, 500]),
        ("Random Forest", False, lambda **kw: RandomForestClassifier(random_state=SEED, **kw), "max_depth", [1, 2, 3, 5, 10, 20, 30]),
        ("Random Forest", False, lambda **kw: RandomForestClassifier(random_state=SEED, **kw), "min_samples_leaf", [1, 3, 5, 10, 30, 80]),
        ("Random Forest", False, lambda **kw: RandomForestClassifier(random_state=SEED, **kw), "max_features", [1, 2, "sqrt", 6, 10]),
        # SVM
        ("SVM", True, lambda **kw: SVC(random_state=SEED, probability=True, **kw), "C", [0.01, 0.1, 1.0, 10.0, 100.0]),
        ("SVM", True, lambda **kw: SVC(random_state=SEED, probability=True, **kw), "gamma", [0.001, 0.01, "scale", 0.5, 2.0]),
        ("SVM", True, lambda **kw: SVC(random_state=SEED, probability=True, **kw), "kernel", ["linear", "poly", "rbf", "sigmoid"]),
        # MLP
        ("MLP", True, lambda **kw: MLPClassifier(random_state=SEED, **kw), "hidden_layer_sizes", [(3,), (15,), (50,), (100,), (200, 100), (300, 150)]),
        ("MLP", True, lambda **kw: MLPClassifier(random_state=SEED, **kw), "learning_rate_init", [0.0001, 0.001, 0.01, 0.1, 0.5]),
        ("MLP", True, lambda **kw: MLPClassifier(random_state=SEED, **kw), "max_iter", [20, 50, 200, 500, 1000]),
        ("MLP", True, lambda **kw: MLPClassifier(random_state=SEED, **kw), "alpha", [0.0001, 0.01, 1.0, 10.0, 50.0]),
    ]

    for tecnica, escalar, builder, hp, valores in pruebas:
        f1s = []
        for val in valores:
            pipe = construir_pipeline(builder(**{hp: val}), num_features, cat_bin_features, escalar=escalar)
            m = evaluar_modelo(pipe, X, y)
            f1s.append(m["F1_Macro"])
            registros.append({
                "Tecnica": tecnica,
                "Hiperparametro": hp,
                "Valor": str(val),
                "F1_Macro": round(m["F1_Macro"], 4),
                "Accuracy": round(m["Accuracy"], 4),
            })
        rango = max(f1s) - min(f1s)
        print(f"  [{tecnica:13s}] {hp:20s} -> min={min(f1s):.4f}, max={max(f1s):.4f} (Δ F1 = {rango:.4f})")

    df_prelim = pd.DataFrame(registros)
    df_prelim.to_csv(os.path.join(OUT_DIR, "tabla_pruebas_preliminares.csv"), index=False)
    print(f"    -> Tabla preliminar guardada: {os.path.join(OUT_DIR, 'tabla_pruebas_preliminares.csv')}")
    return df_prelim


# ==============================================================================
# FASE 3B: EXPERIMENTO OFICIAL (3 HIPERPARÁMETROS x 3 NIVELES = 9 CASOS / TÉCNICA)
# ==============================================================================
def ejecutar_experimento_oficial(X, y, num_features, cat_bin_features):
    print("\n" + "=" * 80)
    print("FASE 3B: DISEÑO EXPERIMENTAL OFICIAL (9 CASOS POR TÉCNICA = 27 CASOS)")
    print("=" * 80)

    # Definición de los 3 experimentos (Bajo / Defecto-Base / Alto para cada uno de los 3 HPs)
    experimentos = {
        "Random Forest": {
            "escalar": False,
            # Base de referencia (n_estimators=100, max_depth=10 estándar RapidMiner, min_samples_leaf=5 de preliminares)
            "base": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5, "random_state": SEED},
            "constructor": RandomForestClassifier,
            "factores": {
                "n_estimators": [("Bajo", 5), ("Defecto", 100), ("Alto", 500)],
                "max_depth": [("Bajo", 1), ("Defecto", 10), ("Alto", 30)],
                "min_samples_leaf": [("Bajo", 1), ("Defecto", 5), ("Alto", 80)],
            },
        },
        "SVM": {
            "escalar": True,
            "base": {"C": 1.0, "gamma": "scale", "kernel": "rbf", "probability": True, "random_state": SEED},
            "constructor": SVC,
            "factores": {
                "C": [("Bajo", 0.01), ("Defecto", 1.0), ("Alto", 100.0)],
                "gamma": [("Bajo", 0.001), ("Defecto", "scale"), ("Alto", 2.0)],
                "kernel": [("Bajo (Lineal)", "linear"), ("Defecto (RBF)", "rbf"), ("Alto (Poli-3)", "poly")],
            },
        },
        "MLP": {
            "escalar": True,
            "base": {"hidden_layer_sizes": (100,), "learning_rate_init": 0.001, "max_iter": 200, "random_state": SEED},
            "constructor": MLPClassifier,
            "factores": {
                "hidden_layer_sizes": [("Bajo", (3,)), ("Defecto", (100,)), ("Alto", (300, 150))],
                "learning_rate_init": [("Bajo", 0.0001), ("Defecto", 0.001), ("Alto", 0.5)],
                "max_iter": [("Bajo", 20), ("Defecto", 200), ("Alto", 1000)],
            },
        },
    }

    tablas_resultados = {}
    mejores_modelos = {}

    for tecnica, cfg in experimentos.items():
        print(f"\n--- Ejecutando 9 casos experimentales para: {tecnica} ---")
        filas = []
        caso_num = 1
        mejor_f1 = -1.0
        mejor_info = None

        for hp_nombre, niveles in cfg["factores"].items():
            for nivel_etiqueta, valor in niveles:
                params = dict(cfg["base"])
                params[hp_nombre] = valor
                modelo = cfg["constructor"](**params)
                pipe = construir_pipeline(modelo, num_features, cat_bin_features, escalar=cfg["escalar"])
                metricas = evaluar_modelo(pipe, X, y)

                fila = {
                    "Caso": f"C{caso_num}",
                    "Hiperparametro": hp_nombre,
                    "Nivel": nivel_etiqueta,
                    "Valor": str(valor),
                    "F1_Macro": round(metricas["F1_Macro"], 4),
                    "F1_Std": round(metricas["F1_Std"], 4),
                    "Accuracy": round(metricas["Accuracy"], 4),
                    "Precision_Macro": round(metricas["Precision_Macro"], 4),
                    "Recall_Macro": round(metricas["Recall_Macro"], 4),
                    "ROC_AUC": round(metricas["ROC_AUC"], 4),
                }
                filas.append(fila)
                print(
                    f"  {fila['Caso']} | {hp_nombre:18s} = {str(valor):10s} ({nivel_etiqueta:13s}) "
                    f"-> F1-Macro: {fila['F1_Macro']:.4f} (+/- {fila['F1_Std']:.4f}) | Acc: {fila['Accuracy']:.4f}"
                )

                if metricas["F1_Macro"] > mejor_f1:
                    mejor_f1 = metricas["F1_Macro"]
                    mejor_info = {
                        "Tecnica": tecnica,
                        "Mejor_Caso": fila["Caso"],
                        "Configuracion_Ganadora": f"{hp_nombre}={valor}",
                        "Params": params,
                        "Escalar": cfg["escalar"],
                        "Constructor": cfg["constructor"],
                        **metricas,
                    }
                caso_num += 1

        df_tec = pd.DataFrame(filas)
        tablas_resultados[tecnica] = df_tec
        mejores_modelos[tecnica] = mejor_info

        nombre_archivo = f"tabla_experimento_{tecnica.replace(' ', '_')}.csv"
        df_tec.to_csv(os.path.join(OUT_DIR, nombre_archivo), index=False)
        graficar_experimento_tecnica(tecnica, df_tec)

    # Generar esquema gráfico de Diseño de Experimentos (Fase 3 / Fase 4)
    generar_esquema_diseno_experimental()

    # Generar tabla comparativa y matrices de confusión de los 3 mejores modelos
    df_comp = generar_comparacion_y_matrices(mejores_modelos, X, y, num_features, cat_bin_features)
    return tablas_resultados, df_comp


def graficar_experimento_tecnica(tecnica, df_tec):
    """Genera una figura con 3 subgráficos (uno por hiperparámetro) mostrando Bajo vs Defecto vs Alto."""
    hps = df_tec["Hiperparametro"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    colores = ["#64748B", "#2563EB", "#0EA5E9"]

    for idx, hp in enumerate(hps):
        sub = df_tec[df_tec["Hiperparametro"] == hp]
        etiquetas = [f"{n}\n{v}" if str(v).startswith("(") else f"{n}\n({v})" for n, v in zip(sub["Nivel"], sub["Valor"])]
        bars = axes[idx].bar(etiquetas, sub["F1_Macro"], yerr=sub["F1_Std"], capsize=5, color=colores, edgecolor="#0F172A", linewidth=0.8)

        # Destacar el valor más alto en verde
        max_idx = sub["F1_Macro"].values.argmax()
        bars[max_idx].set_color("#10B981")

        for b, val, sd in zip(bars, sub["F1_Macro"], sub["F1_Std"]):
            axes[idx].text(b.get_x() + b.get_width() / 2, val + sd + 0.014, f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

        axes[idx].set_title(f"Hiperparámetro: {hp}", fontweight="bold", fontsize=11)
        axes[idx].set_ylim(0.25, 0.90)
        if idx == 0:
            axes[idx].set_ylabel("F1-Score Macro (5-Fold CV)", fontweight="bold")

    fig.suptitle(f"Resultados del Experimento (9 Casos) — {tecnica}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta = os.path.join(FIG_DIR, f"fig_resultados_{tecnica.replace(' ', '_')}.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"    -> Figura de resultados guardada: {ruta}")


def generar_esquema_diseno_experimental():
    """Genera la figura esquemática del Diseño de Experimentos (estilo caja negra: VI -> Técnica -> VD)."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5))

    specs = [
        (
            "Random Forest (RF)",
            ["n_estimators: {5, 100*, 500}", "max_depth: {1, 10*, 30}", "min_samples_leaf: {1, 5*, 80}"],
            "#1E3A8A",
        ),
        (
            "Support Vector Machines (SVM)",
            ["C: {0.01, 1.0*, 100.0}", "gamma: {0.001, scale*, 2.0}", "kernel: {linear, rbf*, poly}"],
            "#0F766E",
        ),
        (
            "Redes Neuronales — MLP",
            ["hidden_layer_sizes: {(3,), (100,)*, (300,150)}", "learning_rate_init: {0.0001, 0.001*, 0.5}", "max_iter (épocas): {20, 200*, 1000}"],
            "#6D28D9",
        ),
    ]

    for ax, (titulo, vars_ind, color) in zip(axes, specs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3.35)
        ax.axis("off")

        # Caja izquierda: Variables Independientes
        box_vi = mpatches.FancyBboxPatch((0.2, 0.25), 3.4, 2.25, boxstyle="round,pad=0.12", facecolor="#F1F5F9", edgecolor=color, linewidth=1.8)
        ax.add_patch(box_vi)
        ax.text(1.9, 2.22, "Variables Independientes (VI)", ha="center", va="center", fontweight="bold", fontsize=10.5, color=color)
        for i, txt in enumerate(vars_ind):
            ax.text(0.4, 1.65 - i * 0.50, f"• {txt}", ha="left", va="center", fontsize=9.2, color="#0F172A")

        # Flecha izquierda -> centro
        ax.annotate("", xy=(4.25, 1.38), xytext=(3.72, 1.38), arrowprops=dict(arrowstyle="->", lw=2.2, color=color))

        # Caja central: Proceso / Modelo
        box_mod = mpatches.FancyBboxPatch((4.3, 0.45), 2.1, 1.85, boxstyle="round,pad=0.12", facecolor=color, edgecolor=color, linewidth=1.5)
        ax.add_patch(box_mod)
        ax.text(5.35, 1.68, titulo, ha="center", va="center", fontweight="bold", fontsize=10.5, color="white")
        ax.text(5.35, 1.05, "Clasificación Binaria\nStratified 5-Fold CV\n(9 casos OFAT)", ha="center", va="center", fontsize=8.8, color="#E2E8F0")

        # Flecha centro -> derecha
        ax.annotate("", xy=(7.05, 1.38), xytext=(6.52, 1.38), arrowprops=dict(arrowstyle="->", lw=2.2, color=color))

        # Caja derecha: Variable Dependiente
        box_vd = mpatches.FancyBboxPatch((7.1, 0.25), 2.7, 2.25, boxstyle="round,pad=0.12", facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.8)
        ax.add_patch(box_vd)
        ax.text(8.45, 2.22, "Variable Dependiente (VD)", ha="center", va="center", fontweight="bold", fontsize=10.5, color="#065F46")
        ax.text(8.45, 1.45, "F1-Score Macro\n(Matriz de Confusión)", ha="center", va="center", fontweight="bold", fontsize=10, color="#0F172A")
        ax.text(8.45, 0.68, "Métricas de apoyo:\nAccuracy, Precision, Recall, AUC", ha="center", va="center", fontsize=8.2, color="#334155")

        # Variables controladas (limpiamente arriba de las cajas)
        ax.text(5.0, 2.98, "Variables Controladas: Dataset CMU Sleep (n=634), Pipeline Imputer+Scaler, Stratified 5-Fold CV (seed=42)  |  * = Valor Defecto/Base", ha="center", va="center", fontsize=8.4, style="italic", color="#475569")

    fig.suptitle("Esquema de Diseño Experimental — Proyecto 3 (Sueño vs. GPA Universitario)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    ruta = os.path.join(FIG_DIR, "fig2_diseno_experimental_esquema.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"\n    -> Esquema de Diseño Experimental guardado: {ruta}")


def generar_comparacion_y_matrices(mejores_modelos, X, y, num_features, cat_bin_features):
    """Genera tabla comparativa final y las 3 matrices de confusión (predicciones out-of-fold)."""
    print("\n" + "=" * 80)
    print("COMPARACIÓN FINAL DE LOS MEJORES MODELOS (MATRICES DE CONFUSIÓN OUT-OF-FOLD)")
    print("=" * 80)

    filas_comp = []
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})

    etiquetas_clase = ["Estándar/Riesgo\n(< 3.5)", "Alto Rend.\n(>= 3.5)"]
    cmaps = ["Blues", "Teal", "Purples"]

    for idx, (tecnica, info) in enumerate(mejores_modelos.items()):
        modelo = info["Constructor"](**info["Params"])
        pipe = construir_pipeline(modelo, num_features, cat_bin_features, escalar=info["Escalar"])
        y_pred = cross_val_predict(pipe, X, y, cv=CV, n_jobs=-1)
        cm = confusion_matrix(y, y_pred)

        f1_m = f1_score(y, y_pred, average="macro")
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="macro")
        rec = recall_score(y, y_pred, average="macro")

        filas_comp.append({
            "Tecnica": tecnica,
            "Mejor_Caso": info["Mejor_Caso"],
            "Hiperparametro_Clave": info["Configuracion_Ganadora"],
            "F1_Macro_CV": round(info["F1_Macro"], 4),
            "F1_Std": round(info["F1_Std"], 4),
            "Accuracy": round(acc, 4),
            "Precision_Macro": round(prec, 4),
            "Recall_Macro": round(rec, 4),
            "ROC_AUC": round(info["ROC_AUC"], 4),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        })

        print(f"\n>>> MEJOR {tecnica} ({info['Mejor_Caso']}: {info['Configuracion_Ganadora']})")
        print(f"    F1-Macro (CV): {info['F1_Macro']:.4f} (+/- {info['F1_Std']:.4f}) | Accuracy: {acc:.4f} | ROC-AUC: {info['ROC_AUC']:.4f}")
        print(f"    Matriz de Confusión [TN={cm[0,0]}, FP={cm[0,1]} | FN={cm[1,0]}, TP={cm[1,1]}]")
        print(classification_report(y, y_pred, target_names=["Estándar/Riesgo (<3.5)", "Alto Rendimiento (>=3.5)"]))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else ("Greens" if idx == 1 else "Purples"),
            cbar=False,
            xticklabels=etiquetas_clase,
            yticklabels=etiquetas_clase,
            annot_kws={"size": 13, "weight": "bold"},
            ax=axes[idx],
        )
        axes[idx].set_title(f"{tecnica}\n({info['Configuracion_Ganadora']})\nF1={info['F1_Macro']:.3f} | Acc={acc:.3f}", fontweight="bold", fontsize=10.5)
        axes[idx].set_xlabel("Predicción")
        if idx == 0:
            axes[idx].set_ylabel("Valor Real")

    df_comp = pd.DataFrame(filas_comp).sort_values("F1_Macro_CV", ascending=False)
    df_comp.to_csv(os.path.join(OUT_DIR, "tabla_comparativa_mejores_modelos.csv"), index=False)

    # Subplot 4: Gráfico comparativo de barras (F1-Macro, Accuracy, ROC-AUC)
    df_bar = df_comp.melt(id_vars="Tecnica", value_vars=["F1_Macro_CV", "Accuracy", "ROC_AUC"], var_name="Metrica", value_name="Puntaje")
    sns.barplot(data=df_bar, x="Tecnica", y="Puntaje", hue="Metrica", palette=["#2563EB", "#10B981", "#F59E0B"], ax=axes[3])
    axes[3].set_title("Comparativa Final de Mejores Modelos", fontweight="bold", fontsize=11)
    axes[3].set_ylim(0.50, 0.90)
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Puntaje (0 - 1)")
    axes[3].legend(loc="lower right", fontsize=8)
    for container in axes[3].containers:
        axes[3].bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    ruta = os.path.join(FIG_DIR, "fig6_comparacion_y_matrices_confusion.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"\n    -> Figura comparativa y matrices guardada: {ruta}")
    print(f"    -> Tabla comparativa final guardada: {os.path.join(OUT_DIR, 'tabla_comparativa_mejores_modelos.csv')}")
    return df_comp


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    df, X, y, num_features, cat_bin_features = cargar_y_limpiar_datos()
    ejecutar_pruebas_preliminares(X, y, num_features, cat_bin_features)
    tablas, df_comp = ejecutar_experimento_oficial(X, y, num_features, cat_bin_features)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL DE LA COMPARACIÓN (ORDENADO POR F1-SCORE MACRO)")
    print("=" * 80)
    print(df_comp[["Tecnica", "Mejor_Caso", "Hiperparametro_Clave", "F1_Macro_CV", "Accuracy", "ROC_AUC"]].to_string(index=False))
    print("\n¡Proceso completo! Todas las tablas CSV y figuras PNG (300 DPI) están en:")
    print(f"  {OUT_DIR}")
