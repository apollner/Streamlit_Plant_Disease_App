import streamlit as st
from PIL import Image
import requests

st.write('# Hello World')
st.write('## Shalom')

#def get_image(url):
    #img = requests.get(url)
    #file = open("sample_image.jpg", "wb")
   # file.write(img.content)
    #file.close()
   # img_file_name = 'sample_image.jpg'
   # return img_file_name



url = st.text_input("Enter Image Url:")

st.image(requests.get('https://upload.wikimedia.org/wikipedia/commons/4/41/Sunflower_from_Silesia2.jpg'))
