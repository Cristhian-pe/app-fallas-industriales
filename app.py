import joblib
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Fallas Industriales", page_icon="⚙️", layout="centered"
)

# Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    data = joblib.load("prototipo_fallas_final.joblib")
    return data["modelo"], data["umbral"], data["columnas"]

try:
    modelo, umbral_optimizado, columnas_modelo = cargar_modelo()
except Exception as e:
    st.error(
        f"No se pudo cargar el archivo del modelo. Asegúrate de ejecutar el script de guardado primero. Error: {e}"
    )
    st.stop()

# Título y descripción
st.title("⚙️ Sistema de Predicción de Fallas en Procesos Industriales")
st.markdown(
    """
Esta aplicación web utiliza un modelo de **Random Forest** entrenado con el dataset **AI4I 2020** (UCI) para clasificar condiciones operativas asociadas con riesgo de falla[cite: 1].
"""
)

st.sidebar.header("🎛️ Parámetros de Operación")

# Entradas del usuario en la barra lateral
tipo_producto = st.sidebar.selectbox("Tipo de Producto", ["L", "M", "H"])
temp_aire = st.sidebar.slider(
    "Temperatura del Aire (K)", 295.0, 310.0, 300.0, 0.1
)
temp_proceso = st.sidebar.slider(
    "Temperatura de Proceso (K)", 305.0, 315.0, 310.0, 0.1
)
velocidad = st.sidebar.slider(
    "Velocidad de Rotación (rpm)", 1100, 3000, 1500, 10
)
torque = st.sidebar.slider("Torque (Nm)", 3.8, 80.0, 40.0, 0.5)
desgaste = st.sidebar.slider("Desgaste de la Herramienta (min)", 0, 250, 120, 1)

# Botón de predicción
if st.sidebar.button("🔍 Evaluar Estado de la Máquina", type="primary"):
  # Crear DataFrame con los datos ingresados
  nuevo_registro = pd.DataFrame(
      [{
          "Air temperature": temp_aire,
          "Process temperature": temp_proceso,
          "Rotational speed": velocidad,
          "Torque": torque,
          "Tool wear": desgaste,
          "Type": tipo_producto,
      }]
  )

  # Asegurar orden correcto de columnas
  nuevo_registro = nuevo_registro[columnas_modelo]

  # Predicción de probabilidad
  probabilidad = float(modelo.predict_proba(nuevo_registro)[0, 1])
  alerta_falla = probabilidad >= umbral_optimizado

  # Mostrar resultados en la interfaz principal
  st.subheader("📊 Resultados del Análisis")

  col1, col2 = st.columns(2)
  with col1:
    st.metric(
        label="Probabilidad Estimada de Falla", value=f"{probabilidad:.1%}"
    )
  with col2:
    st.metric(label="Umbral de Decisión", value=f"{umbral_optimizado:.2f}")

  st.divider()

  if alerta_falla:
    st.error(
        "⚠️ **ALERTA: POSIBLE FALLA DETECTADA**\n\n"
        "Las condiciones operativas actuales superan el umbral de riesgo"
        " establecido. Se recomienda programar una revisión de mantenimiento"
        " preventivo."
    )
  else:
    st.success(
        "✅ **ESTADO OPERATIVO SEGURO**\n\n"
        "No se detectan indicios de falla bajo los parámetros ingresados."
    )

  # Mostrar gráfico o detalles adicionales de los valores ingresados
  with st.expander("Ver detalles de los datos ingresados"):
    st.dataframe(nuevo_registro)

# Nota al pie
st.markdown("---")
st.markdown(
    "<small>Nota: Este prototipo es académico y se basa en datos sintéticos"
    " (AI4I 2020 Predictive Maintenance Dataset). No predice tiempo restante"
    " hasta la falla[cite: 1].</small>",
    unsafe_allow_html=True,
)