import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd
import pytesseract

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

    # Mostrar el texto reconocido
    st.subheader("Texto reconocido en la imagen:")
    if text.strip():
        st.write(text)
    else:
        st.write("No se ha reconocido texto en la imagen.")
