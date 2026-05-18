# =============================================================================
# Morosidad Escolar - Pipeline de Minería de Datos
# Dataset: Colegio San Fernando (dataset_morosidad_colegio.xlsx)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, roc_auc_score, roc_curve,
    classification_report
)

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------
df = pd.read_excel("dataset_morosidad_colegio.xlsx")
print(f"Shape: {df.shape}")
print(df.dtypes)
print(df.head())

# -----------------------------------------------------------------------------
# 2. EXPLORACIÓN INICIAL
# -----------------------------------------------------------------------------
print("\n--- Estadísticas descriptivas ---")
print(df.describe())

print("\n--- Valores nulos ---")
print(df.isnull().sum())

print("\n--- Distribución de la variable target (moroso) ---")
print(df["moroso"].value_counts())
print(df["moroso"].value_counts(normalize=True).round(3))

# -----------------------------------------------------------------------------
# 3. DETECCIÓN DE FUGA DE DATOS (DATA LEAKAGE)
# Variables eliminadas:
#   - id_familia: identificador sin poder predictivo
#   - cuotas_atrasadas_ult_anio: mide comportamiento de morosidad del mismo
#     período que el target → fuga directa
#   - cantidad_recordatorios_ult_anio: el colegio solo manda recordatorios
#     porque la familia ya está en mora → fuga directa
#   - ratio_cuota_ingreso: derivado exacto de cuota_neta_mensual / ingreso
#     → redundancia perfecta, no aporta información nueva
# -----------------------------------------------------------------------------
FEATURES_EXCLUIR = [
    "id_familia",
    "cuotas_atrasadas_ult_anio",
    "cantidad_recordatorios_ult_anio",
    "ratio_cuota_ingreso",
]
TARGET = "moroso"

df_modelo = df.drop(columns=FEATURES_EXCLUIR)
print(f"\nFeatures tras eliminar fugas: {list(df_modelo.columns)}")

# -----------------------------------------------------------------------------
# 4. PREPROCESAMIENTO
# -----------------------------------------------------------------------------
# 4.1 Separar features y target
X = df_modelo.drop(columns=[TARGET])
y = df_modelo[TARGET]

# 4.2 One-Hot Encoding para variables categóricas
cat_cols = X.select_dtypes(include="object").columns.tolist()
print(f"\nColumnas categóricas: {cat_cols}")

X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
print(f"Features tras one-hot encoding: {X.shape[1]}")

# 4.3 División train/test estratificada 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Tasa morosidad train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

# -----------------------------------------------------------------------------
# 5. MODELOS
# -----------------------------------------------------------------------------

# 5.1 Árbol base sin restricciones (para mostrar overfitting)
arbol_base = DecisionTreeClassifier(random_state=42)
arbol_base.fit(X_train, y_train)
acc_base_train = accuracy_score(y_train, arbol_base.predict(X_train))
acc_base_test  = accuracy_score(y_test,  arbol_base.predict(X_test))
print(f"\nÁrbol base (sin restricciones):")
print(f"  Accuracy train: {acc_base_train:.3f} | test: {acc_base_test:.3f}  → overfitting")

# 5.2 CART controlado
cart = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
)
cart.fit(X_train, y_train)

# 5.3 Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
)
rf.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 6. EVALUACIÓN
# -----------------------------------------------------------------------------
for nombre, modelo in [("CART", cart), ("Random Forest", rf)]:
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    print(f"\n=== {nombre} ===")
    print(f"  Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No moroso", "Moroso"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No moroso", "Moroso"])
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{nombre.lower().replace(' ', '_')}.png")
    plt.close()

# ROC Curve
fig, ax = plt.subplots()
for nombre, modelo in [("CART", cart), ("Random Forest", rf)]:
    y_proba = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.set_title("Curva ROC")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# -----------------------------------------------------------------------------
# 7. IMPORTANCIA DE FEATURES (Random Forest)
# -----------------------------------------------------------------------------
importancias = pd.Series(rf.feature_importances_, index=X.columns)
top15 = importancias.nlargest(15)
print("\nTop 15 features más importantes:")
print(top15)

top15.sort_values().plot(kind="barh", title="Importancia de Variables (RF)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# -----------------------------------------------------------------------------
# 8. SIMULACIÓN DE CAMPAÑA DE INTERVENCIÓN
# -----------------------------------------------------------------------------
df_test = X_test.copy()
df_test["moroso_real"] = y_test.values
df_test["prob_moroso"] = rf.predict_proba(X_test)[:, 1]
df_test = df_test.sort_values("prob_moroso", ascending=False).reset_index(drop=True)

n_total       = len(df_test)
total_morosos = df_test["moroso_real"].sum()
presupuesto   = 500
costo_contacto = 2000   # ARS por familia contactada
ahorro_prevencion = 50000  # ARS ahorrado si se previene una morosidad

print(f"\n--- Simulación de intervención (presupuesto: {presupuesto} familias) ---")
top_n = df_test.head(presupuesto)
morosos_captados_modelo = top_n["moroso_real"].sum()
tasa_random = total_morosos / n_total
morosos_captados_random = int(presupuesto * tasa_random)

print(f"  Estrategia aleatoria: {morosos_captados_random} morosos captados")
print(f"  Estrategia modelo:    {morosos_captados_modelo} morosos captados")
print(f"  Mejora: {morosos_captados_modelo - morosos_captados_random} familias adicionales")

# Curva de ganancia acumulada
deciles = np.linspace(0, n_total, 11, dtype=int)
ganancia = [df_test.head(d)["moroso_real"].sum() / total_morosos for d in deciles]
poblacion = [d / n_total for d in deciles]
plt.figure()
plt.plot(poblacion, ganancia, marker="o", label="Modelo RF")
plt.plot([0, 1], [0, 1], "k--", label="Aleatorio")
plt.plot([0, total_morosos/n_total, 1], [0, 1, 1], "g--", alpha=0.4, label="Modelo perfecto")
plt.xlabel("% Familias contactadas")
plt.ylabel("% Morosos captados")
plt.title("Curva de Ganancia Acumulada")
plt.legend()
plt.tight_layout()
plt.savefig("curva_ganancia.png")
plt.close()

print("\nPipeline completo. Gráficos guardados.")
