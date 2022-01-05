import streamlit as st
from PIL import Image
import requests
from io import BytesIO
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
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img)
    

    
else:
    st.write("Paste Image URL")



