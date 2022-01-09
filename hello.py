import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from img_classification import classification

# To predict the image
def predict(image1): 
    model = VGG16()
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label 

st.write('# Hello World')
st.write('## Shalom')
url = st.text_input("Enter Image Url:")
if url:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img)
    classify = st.button("classify image")
    if classify:
        st.write("")
        st.write("Classifying...")
        label = predict(img)
        st.write('%s (%.2f%%)' % (label[1], label[2]*100))
else:
    st.write("Paste Image URL")



