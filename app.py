import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from pathlib import Path
import torch

def map_location(storage, loc):
    return storage

model = load_learner('fruit_model.pkl', cpu=True, map_location=map_location)



# Title
st.title("Fruit Classification")
st.warning(f"This app is trained ONLY specific fruits such as 'Apple,Strawberry,Grape,Tomato,Lemon,Banana,Orange,Peach,Pineapple,Pomegranate and Watermelon'")
st.toggle("I am agree")

#image uploader
file = st.file_uploader("Upload image", type = ['jpg','jpeg','png','svg'])

if file:
    with st.spinner(text="In progress"):
        time.sleep(1)
        st.success("uploaded")

    st.image(file,output_format='JPEG')
    
    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('fruit_model.pkl')

    st.balloons()

    #prediction
    pred, pred_id, probs= model.predict(img)
    st.success(f'Prediction : {pred}')
    st.info(f'Probality : {probs[pred_id]*100:.1f}%')

    #plotting
    figure = px.bar(x = probs * 100, y = model.dls.vocab)
    st.plotly_chart(figure)
