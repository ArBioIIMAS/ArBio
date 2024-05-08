import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import preprocessing
import io
import urllib.request
import os
#****************************************************************#
resize = 512
width = 100
height = 100

def load_logo():
    if not os.path.isfile('logo_arbio.png'):
        urllib.request.urlretrieve('https://arbioiimas.github.io/ArBio/images/logo_arbio.png', 'logo_arbio.png')
    return Image.open('logo_arbio.png')

def add_logo(width, height):
    """Read and return a resized logo"""
    logo = load_logo()
    modified_logo = logo#.resize((width, height))
    return modified_logo


st.title("Deep-cruzi: A tool for segmenting histopathological images based on deep-learning")


# Deep learning-based histopathological segmentation


my_logo = add_logo(width=width, height=height)
st.sidebar.image(my_logo)

#st.sidebar.image(add_logo(logo_path="your/logo/path", width=50, height=60)) 

st.sidebar.markdown("Artificial Intelligence in Biomedicine Group (ArBio)")
st.header("Load an image")



def load_model():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/ArBioIIMAS/ArBio/raw/main/scripts/pesos_chagas.h5', 'model.h5')
    return tf.keras.models.load_model('model.h5')

def main():
    
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    if file_uploaded is not None:
        print("Loading image")
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(fig)

        print("Segmentation")
        model = load_model()
        print(model)
        predictions = predict(model,image)
        st.write(predictions)
        reference = "Reference: Hevia-Montiel, N.; Haro, P.; Guillermo-Cordero, L.; Perez-Gonzalez, J. Deep Learning–Based Segmentation of Trypanosoma cruzi Nests in Histopathological Images. Electronics 2023, 12, 4144. https://doi.org/10.3390/electronics12194144"
        st.write(reference)
   
def predict(model, image):
    IMAGE_SHAPE = (resize, resize,3)
   
    img = image.convert('RGB')
    array_img = np.asarray(img)/255
    x = tf.image.resize(array_img[None, ...],(resize,resize),method='bilinear',antialias=True)
    #mask_array = np.asarray(model.predict(x)[0, ..., 0]*255)

    predictions = model.predict_generator(x, verbose=1)
    mask_array = np.asarray(predictions[0, ..., 0]*255)

    st.header("Nest probability map")
    encode_mask(mask_array)
    st.header("Binary segmentation mask")
    binary_mask(mask_array)

    result = "To save the mask, just right-click on image."
    return result

def encode_mask(mask_array):
    with io.BytesIO() as bimg:
        new_mask = Image.fromarray(mask_array.astype(np.uint8), 'L')
        fig = plt.figure()
        plt.imshow(new_mask)
        plt.axis("off")
        st.pyplot(fig)

def binary_mask(mask_array):
    with io.BytesIO() as bimg:
        import cv2
        r, thresh2 = cv2.threshold(mask_array, 120, 255, cv2.THRESH_BINARY)
        fig = plt.figure()
        plt.imshow(thresh2,cmap="gray")
        plt.axis("off")
        st.pyplot(fig) 

if __name__ == "__main__":
    main()