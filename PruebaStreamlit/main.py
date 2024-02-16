import streamlit as st
from PIL import Image


st.markdown("# Proyecto Final")
st.write('****')
inicio = Image.open('portada.jpeg')
st.image(inicio)
