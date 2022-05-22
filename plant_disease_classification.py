# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:42:50 2022

@author: aronp
"""

# Import all necessary libraries
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
import matplotlib.pyplot as plt
import os
import sys

# This is the list of classes in the order that the model predicts
class_names = ['Apple___Apple_scab',
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

st.write('# Welcome to the leaf/disease predictor')
st.write('### Submit an image of a leaf/disease below either by means of URL link or file (preferably .jpg) from one of the following 38 categories and see it predicts correctly what it is:')

# Function to clean the names of the plants


def plaintxt(name):
    name = name.replace("___", " ")
    name = name.replace("_", " ")
    name = ' '.join(dict.fromkeys(name.split())).capitalize()
    return name


pic_list = []
classes = []

# We get a list of the sample picture paths and a list of the names in the same order to print the togheter in the app
for pic in os.listdir("./leaves_examples/"):
    image = Image.open("./leaves_examples/"+pic)
    pic_list.append(image)
    pic = pic.split(".")[0]
    classes.append(plaintxt(pic))

# We print the images in rows, thanks to having put togheter in lists before
st.image(pic_list, caption=classes, width=100)

st.write('Note that for good results it is necessary to use images taken with similar settings than the examples shown above')
url = st.text_input("Enter Image Url:")
st.write('or')
upload = st.file_uploader("Please Upload Image(JPG/JPEG):")

# Model that was trained on a different repository and saved here
mobilenet_v3 = tf.keras.models.load_model(
    ('mobilenet_v3_large_100_224.h5'), custom_objects={'KerasLayer': hub.KerasLayer})

# Function that process the images and classifies them taking just the one with max probabilities


def classification(img):
    st.write("")
    st.write("Classifying...")
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = mobilenet_v3.predict(img)
    d = prediction.flatten()
    j = d.max()
    for index, item in enumerate(d):
        if item == j:
            class_name = plaintxt(class_names[index])
    confidence = round(100 * j, 3)
    html_str = f"""<style>p.a {{  font: bold 20px sans-serif;}}</style><p class="a">Name: {class_name}</p>"""
    st.markdown(html_str, unsafe_allow_html=True)
    html_str = f"""<style>p.a {{  font: bold 20px sans-serif;}}</style><p class="a">Confidence: {confidence}%</p>"""
    st.markdown(html_str, unsafe_allow_html=True)


# If an URL was given
if url:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        st.image(img)
        classify = st.button("Classify Image from URL")
        if classify:
            classification(img)
        pass
    except:
        st.write('# **Try another link, is preferable that it ends with .jpg**')

# If an image was uploaded
elif upload:
    try:
        content = upload.getvalue()
        bytes_data = upload.read()
        st.image(upload)
        file = Image.open(BytesIO(content))
        classify2 = st.button("classify Image from File")
        if classify2:
            classification(file)
        pass
    except:
        st.write('# **Try another file, is preferable that it ends with .jpg**')

# If an not an image/url was uploaded/submitted
else:
    st.write("Paste Image URL or Upload Image")
