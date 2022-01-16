import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow_hub as hub
import pandas as pd
from matplotlib import pyplot
import os
class_names=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
df = pd.DataFrame(class_names,columns =['Categories'])

mobilenet_v3 = tf.keras.models.load_model(('mobilenet_v3_large_100_224.h5'),custom_objects={'KerasLayer':hub.KerasLayer})

st.write('# Welcome to the PlantVillage leaf disease classifier')
st.write('### The PlantVillage dataset consist of the following 38 categories of plants/diseases:')

pyplot.figure(figsize=(10, 10))

for i,pic in enumerate(os.listdir("./leaves_examples/")):
 image = Image.open("./leaves_examples/"+pic)

 st.image(image, caption=f"{pic}")

 ax = pyplot.subplot(2, 19, i+1)
 pyplot.title(class_names[i])
 pyplot.axis("off")
 #st.write(f"{pic}")
#st.table(df)

st.write('#### Enter a url or upload an image')
st.write('For best results use images showing one leaf like the ones here: https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=plant_village')
url = st.text_input("Enter Image Url:")
st.write('or')
upload = st.file_uploader("Please Upload Image(JPG/JPEG):")
if url:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img)
    classify = st.button("classify image from URL")
    if classify:
        st.write("")
        st.write("Classifying...")
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prediction = mobilenet_v3.predict(img)
        d = prediction.flatten()
        j = d.max()
        for index,item in enumerate(d):
         if item == j:
          class_name = class_names[index]
        confidence = round(100 * j, 3)
        st.write(f"P: {class_name}.\n Confidence: {confidence}%")
elif upload:
  content = upload.getvalue()
  bytes_data = upload.read()
  st.image(upload)
  file = Image.open(BytesIO(content))
  classify2 = st.button("classify image from file")
  if classify2:
        st.write("")
        st.write("Classifying...")
        img = file.resize((224, 224), Image.ANTIALIAS)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prediction = mobilenet_v3.predict(img)
        d = prediction.flatten()
        j = d.max()
        for index,item in enumerate(d):
         if item == j:
          class_name = class_names[index]
        confidence = round(100 * j, 3)
        st.write(f"P: {class_name}.\n Confidence: {confidence}%")
 
else:
    st.write("Paste Image URL or Upload Image")
st.write("**_When using urls from google images beware that not always the image shown corresponds to the actual species you query_**")



