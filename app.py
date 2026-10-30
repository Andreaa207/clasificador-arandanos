
import gradio as gr
import tensorflow as tf
import numpy as np

modelo = tf.keras.models.load_model("modelo_arandanos.keras")

CLASES = ["Verde", "Semimaduro", "Maduro"]

def clasificar(imagen):
    if imagen is None:
        return {}

    imagen = imagen.convert("RGB").resize((128, 128))
    entrada = np.array(imagen, dtype=np.float32)
    entrada = np.expand_dims(entrada, axis=0)

    predicciones = modelo.predict(entrada, verbose=0)[0]

    return {
        CLASES[i]: float(predicciones[i])
        for i in range(3)
    }

app = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Subir imagen del arándano"),
    outputs=gr.Label(num_top_classes=3, label="Resultado"),
    title="Clasificador de maduración de arándanos",
    description="Clasificación mediante una red neuronal convolucional CNN."
)

app.launch()
