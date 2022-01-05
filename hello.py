import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.write('# Hello World')
st.write('## Shalom')
url = st.text_input("Enter Image Url:")
if url:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img)
    
else:
    st.write("Paste Image URL")



