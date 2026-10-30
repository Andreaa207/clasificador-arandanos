
import gradio as gr
import tensorflow as tf
import numpy as np
import os
from huggingface_hub import hf_hub_download

# Descargar el modelo desde Hugging Face
modelo_path = hf_hub_download(
    repo_id="AndreaChilon/modelo-arandanos",
    filename="modelo_arandanos.keras"
)

# Cargar la red neuronal convolucional
modelo = tf.keras.models.load_model(modelo_path)

CLASES = ["Verde", "Semimaduro", "Maduro"]

def clasificar(imagen):
    if imagen is None:
        return {}

    # Preparar la imagen
    imagen = imagen.convert("RGB").resize((128, 128))
    entrada = np.array(imagen, dtype=np.float32)
    entrada = np.expand_dims(entrada, axis=0)

    # Realizar la predicción
    predicciones = modelo.predict(entrada, verbose=0)[0]

    # Mostrar las probabilidades
    return {
        CLASES[i]: float(predicciones[i])
        for i in range(3)
    }

# Interfaz web
app = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(
        type="pil",
        label="Subir imagen del arándano"
    ),
    outputs=gr.Label(
        num_top_classes=3,
        label="Resultado de clasificación"
    ),
    title="Clasificador de maduración de arándanos",
    description="Aplicación basada en una red neuronal convolucional (CNN)."
)

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
