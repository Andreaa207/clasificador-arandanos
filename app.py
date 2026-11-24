import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Clasificador de maduración de arándanos",
    page_icon="🫐",
    layout="centered"
)

CLASES = ["Verde", "Semimaduro", "Maduro"]
REPO_ID = "AndreaChilon/modelo-arandanos"
NOMBRE_MODELO = "modelo_arandanos.keras"

@st.cache_resource
def cargar_modelo():
    ruta_modelo = hf_hub_download(
        repo_id=REPO_ID,
        filename=NOMBRE_MODELO
    )
    return tf.keras.models.load_model(ruta_modelo)

st.title("🫐 Clasificador de maduración de arándanos")
st.write(
    "Aplicación basada en una red neuronal convolucional (CNN) "
    "para identificar la fase de maduración del arándano."
)

with st.spinner("Cargando el modelo de clasificación..."):
    try:
        modelo = cargar_modelo()
    except Exception as error:
        st.error(
            "No se pudo descargar o cargar el modelo desde Hugging Face. "
            "Verifica que el repositorio sea público y que contenga "
            "`modelo_arandanos.keras`."
        )
        st.exception(error)
        st.stop()

st.success("Modelo cargado correctamente.")

archivo = st.file_uploader(
    "Selecciona una fotografía del arándano",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:
    try:
        imagen = Image.open(archivo).convert("RGB")
        st.image(imagen, caption="Imagen seleccionada", use_container_width=True)

        if st.button("Clasificar arándano", type="primary"):
            imagen_procesada = imagen.resize((128, 128))
            entrada = np.array(imagen_procesada, dtype=np.float32)
            entrada = np.expand_dims(entrada, axis=0)

            # La CNN ya incluye Rescaling(1./255).
            # No volver a normalizar la imagen aquí.
            predicciones = modelo.predict(entrada, verbose=0)[0]

            indice = int(np.argmax(predicciones))
            categoria = CLASES[indice]
            confianza = float(predicciones[indice]) * 100

            st.success(f"Clasificación: **{categoria}**")
            st.metric("Confianza de la predicción", f"{confianza:.2f}%")

            st.subheader("Probabilidades por categoría")
            for i, clase in enumerate(CLASES):
                porcentaje = float(predicciones[i]) * 100
                st.write(f"**{clase}:** {porcentaje:.2f}%")
                st.progress(float(predicciones[i]))

    except Exception as error:
        st.error("No se pudo procesar la imagen seleccionada.")
        st.exception(error)
else:
    st.info("Carga una imagen para comenzar la clasificación.")

