import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    roc_curve, classification_report
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Morosidad Escolar - Colegio San Fernando",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

AZUL   = "#1e3a8a"
VERDE  = "#16a34a"
AMBAR  = "#d97706"
ROJO   = "#dc2626"

FEATURES_EXCLUIR = [
    "id_familia",
    "cuotas_atrasadas_ult_anio",
    "cantidad_recordatorios_ult_anio",
    "ratio_cuota_ingreso",
]
TARGET = "moroso"

# =============================================================================
# CARGA Y ENTRENAMIENTO (cacheados)
# =============================================================================
@st.cache_data
def cargar_datos():
    df = pd.read_excel("dataset_morosidad_colegio.xlsx")
    return df

@st.cache_resource
def entrenar_modelos(df):
    df_m = df.drop(columns=FEATURES_EXCLUIR)
    X = df_m.drop(columns=[TARGET])
    y = df_m[TARGET]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    cart = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42,
    )
    cart.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_train, y_train)

    return cart, rf, X_train, X_test, y_train, y_test, X.columns.tolist(), cat_cols

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(f"## 🏫 Morosidad Escolar")
    st.markdown("**Colegio San Fernando**")
    st.divider()
    st.markdown("**Dataset:** 45.000 familias · 23 variables")
    st.markdown("**Modelos:** CART + Random Forest")
    st.markdown("**Target:** `moroso` (0/1)")

# =============================================================================
# CARGA
# =============================================================================
df = cargar_datos()
cart, rf, X_train, X_test, y_train, y_test, feature_names, cat_cols = entrenar_modelos(df)

# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs([
    "🏫 El desafío",
    "👨‍👩‍👧 Las familias",
    "🔍 Los datos",
    "⚠️ La trampa",
    "🤖 El modelo",
    "📈 ¿Qué tan bueno es?",
    "🎯 Probá el modelo",
    "💡 Simulador",
    "✅ Conclusión",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0: EL DESAFÍO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.title("🏫 ¿A qué familia llamar primero?")
    st.markdown(
        "El Colegio San Fernando recauda cuotas de **45.000 familias** cada mes. "
        "El equipo de administración envía recordatorios de pago a todas las familias "
        "con algún atraso, pero los recursos son limitados: llamadas telefónicas, "
        "mensajes, reuniones con directivos."
    )

    col1, col2, col3 = st.columns(3)
    total = len(df)
    morosos = df[TARGET].sum()
    tasa = morosos / total

    col1.metric("Total de familias", f"{total:,}")
    col2.metric("Familias morosas", f"{int(morosos):,}", delta=f"{tasa:.1%} del total", delta_color="inverse")
    col3.metric("Familias al día", f"{int(total - morosos):,}")

    st.divider()
    st.subheader("El problema de actuar sin datos")
    st.markdown(
        f"Si el colegio llama al azar a **100 familias**, en promedio solo "
        f"**{tasa*100:.0f} son morosas** de verdad. El **{(1-tasa)*100:.0f}%** "
        "de los llamados es tiempo y dinero desperdiciado — y peor: familias "
        "al día que reciben un recordatorio innecesario."
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(tasa * 100, 1),
        title={"text": "% de familias morosas"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": ROJO},
            "steps": [
                {"range": [0, 15], "color": "#dcfce7"},
                {"range": [15, 30], "color": "#fef9c3"},
                {"range": [30, 100], "color": "#fee2e2"},
            ],
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Nuestra propuesta:** usar los datos históricos de las familias para "
        "construir un modelo que prediga la probabilidad de morosidad de cada familia "
        "y así **priorizar las intervenciones** donde más impacto van a tener."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: LAS FAMILIAS — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.title("👨‍👩‍👧 ¿Quiénes son las familias?")
    st.markdown("Exploramos el perfil sociodemográfico y financiero del dataset.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Estado civil")
        fig = px.histogram(
            df, x="estado_civil", color="estado_civil",
            title="Distribución por estado civil",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Nivel educativo del referente")
        orden_edu = ["primario", "secundario", "terciario", "universitario", "posgrado"]
        df_edu = df["nivel_educativo_referente"].value_counts().reindex(orden_edu)
        fig = px.bar(
            x=df_edu.index, y=df_edu.values,
            labels={"x": "Nivel", "y": "Cantidad"},
            color=df_edu.index,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tipo de empleo")
        fig = px.pie(
            df, names="tipo_empleo_referente",
            title="Distribución por tipo de empleo",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ingreso mensual familiar")
        fig = px.histogram(
            df, x="ingreso_mensual_familiar", nbins=50,
            color_discrete_sequence=[AZUL],
            labels={"ingreso_mensual_familiar": "Ingreso mensual (ARS)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Morosidad según variables clave")

    col1, col2, col3 = st.columns(3)
    with col1:
        tasa_empleo = df.groupby("tipo_empleo_referente")[TARGET].mean().reset_index()
        tasa_empleo.columns = ["Tipo empleo", "Tasa morosidad"]
        tasa_empleo = tasa_empleo.sort_values("Tasa morosidad", ascending=True)
        fig = px.bar(
            tasa_empleo, x="Tasa morosidad", y="Tipo empleo",
            orientation="h", color="Tasa morosidad",
            color_continuous_scale="RdYlGn_r",
            title="Morosidad por tipo de empleo",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tasa_viv = df.groupby("tipo_vivienda")[TARGET].mean().reset_index()
        tasa_viv.columns = ["Vivienda", "Tasa morosidad"]
        fig = px.bar(
            tasa_viv, x="Vivienda", y="Tasa morosidad",
            color="Tasa morosidad", color_continuous_scale="RdYlGn_r",
            title="Morosidad por tipo de vivienda",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        tasa_pago = df.groupby("medio_pago")[TARGET].mean().reset_index()
        tasa_pago.columns = ["Medio de pago", "Tasa morosidad"]
        fig = px.bar(
            tasa_pago, x="Medio de pago", y="Tasa morosidad",
            color="Tasa morosidad", color_continuous_scale="RdYlGn_r",
            title="Morosidad por medio de pago",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score crediticio vs. Ingreso — por condición de morosidad")
    fig = px.scatter(
        df.sample(3000, random_state=42),
        x="ingreso_mensual_familiar",
        y="score_crediticio",
        color=df.sample(3000, random_state=42)[TARGET].map({0: "No moroso", 1: "Moroso"}),
        color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
        opacity=0.5,
        labels={"ingreso_mensual_familiar": "Ingreso mensual (ARS)", "score_crediticio": "Score crediticio"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: LOS DATOS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.title("🔍 ¿Qué revelan los datos?")

    st.subheader("Calidad del dataset")
    col1, col2 = st.columns(2)

    nulos = df.isnull().sum()
    col1.metric("Valores nulos", int(nulos.sum()))
    col2.metric("Registros completos", f"{len(df):,}")

    st.success("El dataset no tiene valores nulos. Todos los registros están completos.")

    st.subheader("Distribución del target: ¿Está desbalanceado?")
    conteo = df[TARGET].value_counts().reset_index()
    conteo.columns = ["Condición", "Cantidad"]
    conteo["Condición"] = conteo["Condición"].map({0: "No moroso", 1: "Moroso"})
    fig = px.pie(
        conteo, names="Condición", values="Cantidad",
        color="Condición",
        color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
        title="Distribución de la variable objetivo",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "El dataset tiene un **desbalance moderado**: 75% no moroso vs. 25% moroso. "
        "A diferencia de problemas más extremos (95/5), este ratio permite entrenar "
        "modelos razonables, aunque usaremos `class_weight='balanced'` para que el "
        "modelo no ignore la clase minoritaria."
    )

    st.subheader("Distribuciones de variables numéricas clave")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df, x=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            y="score_crediticio",
            color=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
            title="Score crediticio por condición",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            y="pago_a_termino_pct_historico",
            color=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
            title="% pago a término histórico por condición",
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df, x=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            y="ingreso_mensual_familiar",
            color=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
            title="Ingreso mensual familiar por condición",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            y="antiguedad_en_colegio",
            color=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
            title="Antigüedad en el colegio por condición",
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: LA TRAMPA — DATA LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.title("⚠️ La trampa: variables que no deberían estar")
    st.markdown(
        "Antes de entrenar el modelo, hay que revisar si alguna variable "
        "**'mira hacia el futuro'** o **describe directamente el target** — "
        "lo que se llama *data leakage* (fuga de datos)."
    )

    st.error(
        "Si usamos variables que revelan la respuesta antes de conocerla, "
        "el modelo va a parecer muy preciso... pero va a fallar en producción."
    )

    st.subheader("Variables eliminadas por fuga o redundancia")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ❌ `cuotas_atrasadas_ult_anio`")
        st.markdown(
            "Esta variable mide **cuántas cuotas debió en el último año**. "
            "Si el target `moroso=1` se define como 'tiene cuotas impagas', "
            "entonces esta variable **describe directamente el target**. "
            "El modelo aprende a decir 'tiene atrasos = es moroso' sin aprender "
            "los factores de riesgo subyacentes."
        )

        fig = px.box(
            df,
            x=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            y="cuotas_atrasadas_ult_anio",
            color=df[TARGET].map({0: "No moroso", 1: "Moroso"}),
            color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
            title="Cuotas atrasadas vs. condición de morosidad",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### ❌ `cantidad_recordatorios_ult_anio`")
        st.markdown(
            "El colegio solo manda recordatorios **porque la familia ya está en mora**. "
            "Esta variable es consecuencia del target, no predictor."
        )

    with col2:
        st.markdown("#### ❌ `ratio_cuota_ingreso`")
        st.markdown(
            "Este ratio es exactamente `cuota_neta_mensual / ingreso_mensual_familiar`. "
            "Ya tenemos ambas variables en el dataset — "
            "incluirlo sería **redundancia perfecta** que puede distorsionar "
            "la importancia de features sin agregar información nueva."
        )

        st.markdown("#### ❌ `id_familia`")
        st.markdown(
            "Es un identificador único sin poder predictivo. "
            "Si se incluyera, el modelo podría 'memorizar' familias individuales "
            "en lugar de aprender patrones generalizables."
        )

        st.markdown("---")
        st.success(
            "**¿Qué queda?** Un conjunto limpio de **19 variables** que describen "
            "el perfil socioeconómico, financiero y de comportamiento de cada familia "
            "**sin revelar directamente si es morosa o no**."
        )

    st.subheader("Impacto de la trampa: overfitting sin restricciones")
    st.markdown(
        "Para ilustrar el problema, entrenamos un árbol sin límites sobre "
        "el dataset completo (con las variables de fuga incluidas) vs. el dataset limpio."
    )

    arbol_trampa = DecisionTreeClassifier(random_state=42)
    X_trampa = pd.get_dummies(
        df.drop(columns=["id_familia", TARGET]),
        columns=cat_cols, drop_first=False,
    )
    y_trampa = df[TARGET]
    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(
        X_trampa, y_trampa, test_size=0.30, random_state=42, stratify=y_trampa
    )
    arbol_trampa.fit(X_tr_t, y_tr_t)
    acc_train_trampa = accuracy_score(y_tr_t, arbol_trampa.predict(X_tr_t))
    acc_test_trampa  = accuracy_score(y_te_t, arbol_trampa.predict(X_te_t))

    arbol_limpio = DecisionTreeClassifier(random_state=42)
    X_limpio_train = pd.DataFrame(X_train, columns=feature_names)
    arbol_limpio.fit(X_train, y_train)
    acc_train_limpio = accuracy_score(y_train, arbol_limpio.predict(X_train))
    acc_test_limpio  = accuracy_score(y_test,  arbol_limpio.predict(X_test))

    comp = pd.DataFrame({
        "Dataset": ["Con variables de fuga", "Sin variables de fuga"],
        "Accuracy TRAIN": [acc_train_trampa, acc_train_limpio],
        "Accuracy TEST":  [acc_test_trampa,  acc_test_limpio],
    })

    col1, col2 = st.columns(2)
    col1.dataframe(comp.set_index("Dataset").style.format("{:.3f}"), use_container_width=True)
    col2.markdown(
        f"Con las variables de fuga, el árbol sin restricciones logra "
        f"**{acc_train_trampa:.1%} en entrenamiento** pero solo "
        f"**{acc_test_trampa:.1%} en test**. "
        "El modelo memorizó el dataset en lugar de aprender."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: EL MODELO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.title("🤖 Construyendo el modelo predictivo")

    st.subheader("División del dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Entrenamiento (70%)", f"{len(X_train):,} familias")
    col2.metric("Prueba (30%)", f"{len(X_test):,} familias")
    col3.metric("Variables predictoras", len(feature_names))

    st.markdown(
        "La división es **estratificada**: tanto en entrenamiento como en test "
        "se mantiene la proporción original de familias morosas (25%)."
    )

    st.subheader("Dos modelos, dos filosofías")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌳 CART — Árbol de decisión")
        st.markdown(
            "Un único árbol que aprende reglas de decisión. "
            "**Ventaja:** muy interpretable — se puede seguir cada decisión. "
            "**Desventaja:** sensible al ruido en los datos."
        )
        st.code("""DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
)""")
        st.markdown(
            "- `max_depth=5`: máximo 5 niveles de profundidad → evita memorización\n"
            "- `min_samples_leaf=20`: cada hoja necesita al menos 20 familias\n"
            "- `class_weight='balanced'`: penaliza más los errores en morosos\n"
        )

    with col2:
        st.markdown("### 🌲 Random Forest — Bosque aleatorio")
        st.markdown(
            "100 árboles diferentes, cada uno entrenado sobre una muestra aleatoria "
            "del dataset y con un subconjunto aleatorio de variables. "
            "La predicción final es la **votación de todos los árboles**."
        )
        st.code("""RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
)""")
        st.markdown(
            "- `n_estimators=100`: 100 árboles distintos\n"
            "- Mismos parámetros de control que CART para comparación justa\n"
            "- Reduce overfitting por promediado de predicciones\n"
        )

    st.subheader("Preprocesamiento de variables")
    st.markdown(
        "Las variables categóricas se codifican con **One-Hot Encoding** "
        "(no LabelEncoder) para evitar introducir relaciones de orden falsas. "
        "Por ejemplo: `tipo_empleo = formal` no es 'mayor' que `informal`."
    )
    st.markdown(f"Variables categóricas codificadas: `{'`, `'.join(cat_cols)}`")
    st.markdown(f"Total de features tras encoding: **{len(feature_names)}**")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.title("📈 ¿Qué tan bueno es el modelo?")

    y_pred_cart  = cart.predict(X_test)
    y_pred_rf    = rf.predict(X_test)
    y_proba_cart = cart.predict_proba(X_test)[:, 1]
    y_proba_rf   = rf.predict_proba(X_test)[:, 1]

    acc_cart = accuracy_score(y_test, y_pred_cart)
    acc_rf   = accuracy_score(y_test, y_pred_rf)
    auc_cart = roc_auc_score(y_test, y_proba_cart)
    auc_rf   = roc_auc_score(y_test, y_proba_rf)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CART — Accuracy", f"{acc_cart:.1%}")
    col2.metric("CART — AUC-ROC",  f"{auc_cart:.3f}")
    col3.metric("RF — Accuracy",   f"{acc_rf:.1%}")
    col4.metric("RF — AUC-ROC",    f"{auc_rf:.3f}")

    st.markdown(
        "**¿Por qué AUC-ROC y no solo accuracy?** Con un 75/25 de desbalance, "
        "un modelo que predice 'nadie es moroso' lograría 75% de accuracy — y sería inútil. "
        "El AUC-ROC mide la **capacidad de discriminar** entre morosos y no morosos "
        "independientemente del umbral de clasificación."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Curva ROC")
        fig = go.Figure()
        for nombre, fpr_tpr in [
            ("CART", roc_curve(y_test, y_proba_cart)),
            ("Random Forest", roc_curve(y_test, y_proba_rf)),
        ]:
            fpr, tpr, _ = fpr_tpr
            auc = roc_auc_score(y_test, y_proba_cart if nombre == "CART" else y_proba_rf)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{nombre} (AUC={auc:.3f})", mode="lines"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Aleatorio", line=dict(dash="dash", color="gray")))
        fig.update_layout(
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Matrices de confusión")
        modelo_sel = st.radio("Seleccioná el modelo:", ["CART", "Random Forest"], horizontal=True)
        y_pred_sel = y_pred_cart if modelo_sel == "CART" else y_pred_rf
        cm = confusion_matrix(y_test, y_pred_sel)
        labels = ["No moroso", "Moroso"]
        fig = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale="Blues",
            labels={"x": "Predicción", "y": "Real"},
            title=f"Matriz de confusión — {modelo_sel}",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Importancia de variables — Random Forest")
    importancias = pd.Series(rf.feature_importances_, index=feature_names)
    top15 = importancias.nlargest(15).sort_values()
    fig = px.bar(
        x=top15.values, y=top15.index, orientation="h",
        color=top15.values, color_continuous_scale="Blues",
        labels={"x": "Importancia (Gini)", "y": "Variable"},
        title="Top 15 variables más predictivas",
    )
    fig.update_layout(coloraxis_showscale=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reporte de clasificación completo")
    reporte = classification_report(
        y_test, y_pred_rf,
        target_names=["No moroso", "Moroso"],
        output_dict=True,
    )
    df_rep = pd.DataFrame(reporte).T.round(3)
    st.dataframe(df_rep, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: PROBÁ EL MODELO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.title("🎯 Probá el modelo con una familia nueva")
    st.markdown("Completá el perfil de la familia y el modelo estimará su probabilidad de morosidad.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Perfil familiar**")
        cantidad_hijos      = st.slider("Cantidad de hijos", 1, 5, 2)
        hijos_en_colegio    = st.slider("Hijos en el colegio", 1, 5, 1)
        edad_referente      = st.slider("Edad del referente", 24, 70, 40)
        estado_civil        = st.selectbox("Estado civil", ["casado", "soltero", "divorciado", "union_convivencial", "viudo"])
        ambos_trabajan      = st.radio("¿Ambos padres trabajan?", [1, 0], format_func=lambda x: "Sí" if x else "No", horizontal=True)

    with col2:
        st.markdown("**Perfil laboral y educativo**")
        nivel_edu   = st.selectbox("Nivel educativo del referente", ["primario", "secundario", "terciario", "universitario", "posgrado"])
        tipo_empleo = st.selectbox("Tipo de empleo", ["formal", "independiente", "informal", "desempleado"])
        ingreso     = st.number_input("Ingreso mensual familiar (ARS)", min_value=180_000, max_value=5_500_000, value=700_000, step=50_000)
        deuda       = st.number_input("Deuda en otros créditos (ARS)", min_value=0, max_value=7_000_000, value=200_000, step=50_000)
        score       = st.slider("Score crediticio", 300, 850, 620)

    with col3:
        st.markdown("**Perfil de pago y ubicación**")
        cuota_mensual   = st.number_input("Cuota mensual (ARS)", min_value=60_000, max_value=1_500_000, value=200_000, step=10_000)
        beca_pct        = st.selectbox("Descuento/beca (%)", [0, 10, 15, 20, 30, 50])
        tipo_vivienda   = st.selectbox("Tipo de vivienda", ["propia", "alquilada", "familiar"])
        medio_pago      = st.selectbox("Medio de pago", ["debito_automatico", "transferencia", "efectivo"])
        antiguedad      = st.slider("Antigüedad en el colegio (años)", 0, 14, 3)
        distancia       = st.number_input("Distancia al colegio (km)", min_value=0.3, max_value=45.0, value=5.0, step=0.5)
        zona            = st.selectbox("Zona", ["centro", "norte", "sur", "este", "oeste"])
        pago_historico  = st.slider("% pago a término histórico", 0, 100, 75)

    if st.button("🔍 Predecir morosidad", type="primary"):
        cuota_neta = int(cuota_mensual * (1 - beca_pct / 100))

        nueva_fam = pd.DataFrame([{
            "cantidad_hijos": cantidad_hijos,
            "hijos_en_colegio": hijos_en_colegio,
            "edad_referente": edad_referente,
            "estado_civil": estado_civil,
            "nivel_educativo_referente": nivel_edu,
            "tipo_empleo_referente": tipo_empleo,
            "ambos_padres_trabajan": ambos_trabajan,
            "ingreso_mensual_familiar": ingreso,
            "cuota_mensual": cuota_mensual,
            "becas_descuento_pct": beca_pct,
            "cuota_neta_mensual": cuota_neta,
            "tipo_vivienda": tipo_vivienda,
            "antiguedad_en_colegio": antiguedad,
            "medio_pago": medio_pago,
            "distancia_colegio_km": distancia,
            "zona": zona,
            "deuda_otros_creditos": deuda,
            "score_crediticio": score,
            "pago_a_termino_pct_historico": pago_historico,
        }])

        nueva_enc = pd.get_dummies(nueva_fam, columns=cat_cols, drop_first=False)
        for col in feature_names:
            if col not in nueva_enc.columns:
                nueva_enc[col] = 0
        nueva_enc = nueva_enc[feature_names]

        prob = rf.predict_proba(nueva_enc)[0][1]
        pred = rf.predict(nueva_enc)[0]

        col1, col2 = st.columns(2)
        col1.metric(
            "Probabilidad de morosidad",
            f"{prob:.1%}",
            delta=None,
        )
        col2.metric(
            "Clasificación",
            "⚠️ MOROSO" if pred == 1 else "✅ AL DÍA",
        )

        color_gauge = ROJO if prob > 0.5 else (AMBAR if prob > 0.3 else VERDE)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color_gauge},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 50},
            },
            number={"suffix": "%"},
            title={"text": "Probabilidad de morosidad"},
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: SIMULADOR DE INTERVENCIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.title("💡 Simulador de campaña de intervención")
    st.markdown(
        "¿Cuánto mejora el resultado si el equipo de administración contacta "
        "a las familias ordenadas por probabilidad de morosidad en lugar de al azar?"
    )

    col1, col2, col3 = st.columns(3)
    presupuesto      = col1.slider("Familias a contactar", 100, len(y_test), 1000, step=100)
    costo_contacto   = col2.number_input("Costo por contacto (ARS)", min_value=500, max_value=20_000, value=3_000, step=500)
    ahorro_prevencion = col3.number_input("Ahorro por morosidad evitada (ARS)", min_value=10_000, max_value=500_000, value=80_000, step=10_000)

    df_test_sim = pd.DataFrame(X_test, columns=feature_names).copy()
    df_test_sim["moroso_real"]  = y_test.values
    df_test_sim["prob_moroso"]  = rf.predict_proba(X_test)[:, 1]
    df_test_sim = df_test_sim.sort_values("prob_moroso", ascending=False).reset_index(drop=True)

    n_total       = len(df_test_sim)
    total_morosos = int(df_test_sim["moroso_real"].sum())
    tasa_base     = total_morosos / n_total

    top_n = df_test_sim.head(presupuesto)
    m_modelo  = int(top_n["moroso_real"].sum())
    m_random  = int(round(presupuesto * tasa_base))

    roi_modelo  = (m_modelo * ahorro_prevencion - presupuesto * costo_contacto) / max(presupuesto * costo_contacto, 1)
    roi_random  = (m_random * ahorro_prevencion - presupuesto * costo_contacto) / max(presupuesto * costo_contacto, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Morosos captados (modelo)", m_modelo,   delta=f"+{m_modelo - m_random} vs aleatorio")
    col2.metric("Morosos captados (azar)",   m_random)
    col3.metric("ROI estrategia modelo",     f"{roi_modelo:.1%}")
    col4.metric("ROI estrategia azar",       f"{roi_random:.1%}")

    st.subheader("Curva de ganancia acumulada")
    st.markdown(
        "Esta curva muestra: si contactamos el X% de las familias "
        "(ordenadas por probabilidad), ¿qué % de todos los morosos capturamos?"
    )

    pasos    = np.linspace(0, n_total, 200, dtype=int)
    ganancia = [df_test_sim.head(p)["moroso_real"].sum() / total_morosos for p in pasos]
    porc_pob = [p / n_total for p in pasos]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p * 100 for p in porc_pob],
        y=[g * 100 for g in ganancia],
        name="Modelo (Random Forest)", line=dict(color=AZUL, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        name="Estrategia aleatoria", line=dict(dash="dash", color="gray"),
    ))
    fig.add_trace(go.Scatter(
        x=[0, tasa_base * 100, 100], y=[0, 100, 100],
        name="Modelo perfecto", line=dict(dash="dot", color=VERDE, width=1),
        opacity=0.5,
    ))
    fig.add_vline(
        x=presupuesto / n_total * 100,
        line_dash="dash", line_color=AMBAR,
        annotation_text=f"Tu presupuesto ({presupuesto} familias)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="% de familias contactadas",
        yaxis_title="% de morosos captados",
        height=450,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribución de probabilidades en el test set")
    fig = px.histogram(
        df_test_sim, x="prob_moroso",
        color=df_test_sim["moroso_real"].map({0: "No moroso", 1: "Moroso"}),
        color_discrete_map={"No moroso": VERDE, "Moroso": ROJO},
        nbins=40, barmode="overlay", opacity=0.7,
        labels={"prob_moroso": "Probabilidad predicha de morosidad"},
        title="¿Cómo separa el modelo a morosos de no morosos?",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8: CONCLUSIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.title("✅ Conclusión y próximos pasos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lo que aprendimos")
        st.markdown("""
**Del análisis exploratorio:**
- Las familias con empleo informal o desempleadas tienen tasas de morosidad significativamente mayores
- El score crediticio y el historial de pago son señales claras de riesgo
- El medio de pago (efectivo vs. débito automático) también diferencia perfiles

**Del modelado:**
- Un árbol sin restricciones memoriza el dataset (overfitting)
- El Random Forest con `class_weight='balanced'` es el mejor modelo: mayor AUC y más robusto
- Las variables más predictivas son el historial de pago, el score crediticio y el tipo de empleo

**De la detección de leakage:**
- `cuotas_atrasadas_ult_anio` y `cantidad_recordatorios_ult_anio` describen el target → eliminadas
- `ratio_cuota_ingreso` es redundante con `cuota_neta` e `ingreso` → eliminada
        """)

    with col2:
        st.subheader("Impacto de negocio")
        st.markdown("""
**Comparación de estrategias:**
- Estrategia aleatoria: ~25% de eficiencia (tasa base de morosidad)
- Estrategia modelo: capta en los primeros 30% de familias contactadas más del 60% de morosos

**Próximos pasos recomendados:**
1. **Validar con datos temporales**: entrenar en períodos anteriores, predecir el período siguiente
2. **Ajustar el umbral de clasificación** según el costo real del error de tipo I vs. tipo II
3. **Incorporar variables externas**: índices de inflación, estacionalidad, situación económica
4. **Modelo de intervención temprana**: en lugar de predecir quién está moroso, predecir quién va a caer en mora en los próximos 3 meses
5. **Segmentación**: familias de riesgo alto → llamada telefónica; riesgo medio → mensaje automático

> El modelo no reemplaza el juicio del equipo de administración — le da información para actuar mejor.
        """)

    st.divider()
    col1, col2, col3 = st.columns(3)
    y_proba_rf_full = rf.predict_proba(X_test)[:, 1]
    auc_final  = roc_auc_score(y_test, y_proba_rf_full)
    acc_final  = accuracy_score(y_test, rf.predict(X_test))
    mejora_sim = int(rf.predict_proba(X_test)[:, 1].sort_values().index[::-1][:1000] if False else 0)

    col1.metric("AUC-ROC final (RF)", f"{auc_final:.3f}")
    col2.metric("Accuracy final (RF)", f"{acc_final:.1%}")
    col3.metric("Variables en el modelo", len(feature_names))

    st.markdown("---")
    st.caption(
        "Proyecto Integrador — Minería de Datos Empresariales | CAECE 2026 | "
        "Dataset: Colegio San Fernando (simulado con fines educativos)"
    )
