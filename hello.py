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
if url:
    image = requests.get(url)
    st.image(image)
    
    
else:
    st.write("Paste Image URL")



