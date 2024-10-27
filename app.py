import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd
import pytesseract
from gtts import gTTS
from googletrans import Translator
import os
import time
import glob

# Configuración inicial
text = ""
translator = Translator()

def text_to_speech(input_language, output_language, text, tld):
    translation = translator.translate(text, src=input_language, dest=output_language)
    trans_text = translation.text
    tts = gTTS(trans_text, lang=output_language, tld=tld, slow=False)
    try:
        my_file_name = text[:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, trans_text

def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if mp3_files:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)

# Crear directorio temporal si no existe
try:
    os.mkdir("temp")
except FileExistsError:
    pass

remove_files(7)

# load pretrained model
model = yolov5.load('yolov5s.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

st.title("Detección de Objetos e Imágenes")

# Sidebar para parámetros
with st.sidebar:
    st.subheader('Parámetros de Configuración')
    model.iou = st.slider('Seleccione el IoU', 0.0, 1.0)
    st.write('IOU:', model.iou)

with st.sidebar:
    model.conf = st.slider('Seleccione el Confidence', 0.0, 1.0)
    st.write('Conf:', model.conf)
    
    st.subheader("Procesamiento para Cámara")
    filtro = st.radio("Filtro para imagen con cámara", ('Sí', 'No'))

# Crear pestañas para separar las opciones de entrada
tab1, tab2 = st.tabs(["Cámara", "Subir Archivo"])

# Advertencia inicial que se mostrará si no hay texto reconocido
warning_message = st.empty()
loading_placeholder = st.sidebar.empty()

with tab1:
    picture = st.camera_input("Capturar foto", label_visibility='visible')
    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if filtro == 'Sí':
            cv2_img = cv2.bitwise_not(cv2_img)
        process_image = True
    else:
        process_image = False

with tab2:
    uploaded_file = st.file_uploader("Seleccionar imagen", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        process_image = True
    else:
        process_image = False

# Procesar la imagen si hay una imagen disponible
if process_image:
    # Object detection
    results = model(cv2_img)
    predictions = results.pred[0]
    boxes = predictions[:, :4] 
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    
    col1, col2 = st.columns(2)
    
    with col1:
        results.render()
        st.image(cv2_img, channels='BGR')
    
    with col2:      
        label_names = model.names
        
        if len(categories) > 0:
            category_count = {}
            for category in categories:
                if category in category_count:
                    category_count[category] += 1
                else:
                    category_count[category] = 1        
            
            data = []        
            for category, count in category_count.items():
                label = label_names[int(category)]            
                data.append({"Categoría": label, "Cantidad": count})
            
            data2 = pd.DataFrame(data)
            df_sum = data2.groupby('Categoría')['Cantidad'].sum().reset_index()
            st.write("Objetos detectados:")
            st.write(df_sum)
        else:
            st.write("No se detectaron objetos en la imagen.")
    
    # OCR Processing
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img_rgb)

    # Solo mostrar los parámetros de traducción si se ha reconocido texto
    if text.strip():
        detected_language = translator.detect(text).lang
        warning_message.empty()
        
        with st.sidebar:
            st.subheader("Parámetros de traducción")
            
            lang_map = {
                "en": "Inglés",
                "es": "Español",
                "bn": "Bengalí",
                "ko": "Coreano",
                "zh-cn": "Mandarín",
                "ja": "Japonés",
                "fr": "Francés",
                "de": "Alemán",
                "pt": "Portugués",
                "ru": "Ruso"
            }

            st.markdown("### Texto reconocido:")
            st.write(text)
            st.markdown(f"**Idioma detectado:** {lang_map.get(detected_language, 'Desconocido')}")

            in_lang_name = lang_map.get(detected_language, "Desconocido")
            in_lang_options = list(lang_map.values())
            in_lang_index = in_lang_options.index(in_lang_name) if in_lang_name in in_lang_options else 0

            st.selectbox("Seleccione el lenguaje de entrada", in_lang_options, index=in_lang_index)

            out_lang = st.selectbox(
                "Selecciona tu idioma de salida",
                tuple(lang_map.values())
            )

            output_language = {v: k for k, v in lang_map.items()}[out_lang]

            english_accent = st.selectbox(
                "Seleccione el acento (solo aplica para inglés)",
                (
                    "Default",
                    "India",
                    "United Kingdom",
                    "United States",
                    "Canada",
                    "Australia",
                    "Ireland",
                    "South Africa",
                )
            )
            
            tld = {
                "Default": "com",
                "India": "co.in",
                "United Kingdom": "co.uk",
                "United States": "com",
                "Canada": "ca",
                "Australia": "com.au",
                "Ireland": "ie",
                "South Africa": "co.za",
            }[english_accent]

            if st.button("Convertir"):
                loading_placeholder.image("dog.gif")
                
                result, output_text = text_to_speech(detected_language, output_language, text, tld)
                audio_file = open(f"temp/{result}.mp3", "rb")
                audio_bytes = audio_file.read()
                st.markdown("## Tu audio:")
                st.audio(audio_bytes, format="audio/mp3", start_time=0)

                loading_placeholder.empty()

                st.markdown("## Texto traducido:")
                st.write(output_text)
    else:
        warning_message.warning("No se ha reconocido texto en la imagen.")
