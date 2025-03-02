import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langdetect import detect

# Configura el modelo y el tokenizador
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "PlanTL-GOB-ES/gpt2-base-bne"  # Modelo en español
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Función para detectar el idioma
def detectar_idioma(texto):
    try:
        return detect(texto)
    except:
        return "es"  # Por defecto, español

# Función para generar respuestas
def generar_respuesta(mensaje, historial=[]):
    idioma = detectar_idioma(mensaje)
    
    # Ajusta el prompt según el idioma
    if idioma == "es":
        prompt = f"Usuario: {mensaje}\nAsistente:"
    else:
        prompt = f"User: {mensaje}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=2.0  # Penalización por repetición
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n")[-1]
    historial.append((mensaje, respuesta))
    return historial

# Interfaz personalizada con Gradio Blocks
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 CIBERNEXUS - Chatbot Multilingüe")
    
    # Área de publicidad
    gr.HTML("""
    <div style="text-align: center; padding: 10px; background-color: #f0f0f0; border-radius: 10px;">
        <h3>Espacio Publicitario</h3>
        <p>Contáctanos para anunciarte aquí.</p>
    </div>
    """)
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversación", height=500)  # Aumenta la altura
        with gr.Column():
            entrada = gr.Textbox(label="Escribe tu mensaje", placeholder="Hola, ¿cómo estás?")
            boton = gr.Button("Enviar 🚀")
    boton.click(generar_respuesta, entrada, chatbot)

# Ejecuta la aplicación
demo.launch()