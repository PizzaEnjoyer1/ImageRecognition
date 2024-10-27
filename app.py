import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd
import easyocr

# Inicializar el lector OCR (solo se hace una vez)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['es', 'en'])  # Soporta español e inglés

reader = load_ocr()

# load pretrained model
model = yolov5.load('yolov5s.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

st.title("Detección de Objetos en Imágenes")

# Sidebar para parámetros
with st.sidebar:
    st.subheader('Parámetros de Configuración')
    model.iou = st.slider('Seleccione el IoU', 0.0, 1.0)
    st.write('IOU:', model.iou)

with st.sidebar:
    model.conf = st.slider('Seleccione el Confidence', 0.0, 1.0)
    st.write('Conf:', model.conf)

# Crear pestañas para separar las opciones de entrada
tab1, tab2 = st.tabs(["Cámara", "Subir Archivo"])

with tab1:
    picture = st.camera_input("Capturar foto", label_visibility='visible')
    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
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

# Procesar la imagen si hay una imagen disponible (ya sea de cámara o archivo)
if process_image:
    # perform inference for object detection
    results = model(cv2_img)
    
    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] 
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    
    # Crear tres columnas
    col1, col2 = st.columns(2)
    
    with col1:
        # show detection bounding boxes on image
        results.render()
        # show image with detections 
        st.image(cv2_img, channels='BGR')
    
    with col2:      
        # get label names
        label_names = model.names
        
        if len(categories) > 0:  # Verificar si hay objetos detectados
            # count categories
            category_count = {}
            for category in categories:
                if category in category_count:
                    category_count[category] += 1
                else:
                    category_count[category] = 1        
            
            data = []        
            # print category counts and labels
            for category, count in category_count.items():
                label = label_names[int(category)]            
                data.append({"Categoría": label, "Cantidad": count})
            
            data2 = pd.DataFrame(data)
            
            # agrupar los datos por la columna "categoria" y sumar las cantidades
            df_sum = data2.groupby('Categoría')['Cantidad'].sum().reset_index()
            st.write("Objetos detectados:")
            st.write(df_sum)
        else:
            st.write("No se detectaron objetos en la imagen.")
    
    # Realizar OCR en la imagen
    st.subheader("Texto detectado en la imagen:")
    with st.spinner('Detectando texto...'):
        results = reader.readtext(cv2_img)
        
        if results:
            # Crear una tabla para mostrar el texto detectado
            text_data = []
            for detection in results:
                text = detection[1]  # El texto detectado
                conf = detection[2]  # La confianza de la detección
                text_data.append({"Texto": text, "Confianza": f"{conf:.2%}"})
            
            text_df = pd.DataFrame(text_data)
            st.write(text_df)
        else:
            st.write("No se detectó texto en la imagen.")
