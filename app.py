import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configura el dispositivo (GPU si está disponible, si no, CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

# Usamos un modelo público en español
MODEL_NAME = "bertin-project/bertin-gpt-j-6B"  # Modelo en español, gratuito y público
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Función del Chatbot
def generar_respuesta_lujo(mensaje, historial):
    # Prepara el prompt
    prompt = f"""Eres CYBERNEXUS-AI, un asistente virtual profesional en español. Responde de forma clara, breve y útil.
    Siempre mantén un tono formal y amable.
    Si no entiendes algo, pide más detalles.
    No inventes información ni des respuestas cortantes.

    Usuario: {mensaje}
    Asistente:"""

    # Tokeniza el prompt y genera la respuesta
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # Limita la longitud de la respuesta para ahorrar RAM
        do_sample=True,
        temperature=0.1,  # Reduce la temperatura para más coherencia
        top_p=0.9,  # Filtra las opciones menos probables
        repetition_penalty=2.0,  # Aumenta la penalización por repeticiones
        num_beams=1  # Usa búsqueda greedy para ahorrar RAM
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Asistente:")[-1]

    # Añade la respuesta al historial
    historial.append((mensaje, respuesta))  # Formato: [(usuario, asistente), ...]

    # Devuelve el historial actualizado
    return historial

# Interfaz de Gradio
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="CONSOLE")
    entrada = gr.Textbox(label="COMANDO", placeholder="Escribe tu consulta", lines=1)
    gr.Button("ENVIAR 🚀").click(generar_respuesta_lujo, [entrada, chatbot], chatbot)

# Usa la variable de entorno PORT, o 7860 si no está definida
PORT = int(os.environ.get("PORT", 7860))

# Lanza la aplicación
demo.launch(server_name="0.0.0.0", server_port=PORT)